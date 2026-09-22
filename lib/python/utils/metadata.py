"""Per-step run metadata, written as one JSON document per step invocation.

Every step emits a companion record describing what it read, what it wrote,
with which checksums, under which code and which tool versions. Many runs can
then be loaded and queried together::

    SELECT step, dataset, status, duration_s
    FROM read_json_auto('<root>/*/results/metadata/**/*.json', union_by_name=true)
    ORDER BY started_at DESC;

Schema notes, all of them chosen so that heterogeneous steps union cleanly in
DuckDB rather than producing a ragged table:

- Top level is flat scalars plus a few ``LIST<STRUCT>`` columns. Scalars stay
  queryable without unnesting.
- ``params`` is a list of ``{key, value}`` with ``value`` always a **string**.
  A plain object would give every step a different STRUCT and the union would
  be a mess. ``command`` keeps the full argv as ground truth; ``params`` is the
  curated subset worth querying.
- ``inputs`` and ``outputs`` share one shape, so they can be UNIONed into a
  single "files" view.
- Who ran it is recorded at two levels, both as flat scalars: ``user``/``uid``
  is the cluster account, ``git_user_name``/``git_user_email`` is the person
  from git config. They differ whenever a lab account is shared, and the git
  identity is the one that survives an opaque account name.

Checksums
---------
md5 is on by default and streamed in 8 MiB chunks. The expensive case is a
multi-GB fragments file, and that is precisely the step whose runtime already
dwarfs the hash. Set ``METADATA_CHECKSUMS=0`` to skip; the record then
carries ``md5: null`` with ``md5_skipped: "disabled"``, which is distinguishable
from a file that was simply absent.

Granularity
-----------
A numbered step name (``00.1.preprocess_peaks``) is one *sbatch job*, written by
the EXIT trap in lib/bash/common.sh. An unnumbered one (``preprocess_peaks``) is
one *tool invocation*, written by the Python script itself -- a job that loops
over datasets produces one job record and several tool records. The names are
kept distinct on purpose so the two granularities coexist in the same table
rather than double-counting; filter on ``step LIKE '__.%'`` for jobs only.

Failure
-------
Outputs are declared up front but hashed on exit, so a run that dies still
records what it *intended* to produce, with ``exists: false``. The record is
written for failed runs too (``status: "failed"`` plus ``error``). A step killed
with SIGKILL (SLURM OOM or hard preemption) cannot write anything -- the absence
of a record is itself the signal.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from utils import log

logger = log.get_logger(__name__)

SCHEMA_VERSION = 1

CHECKSUM_ENV_VAR = "METADATA_CHECKSUMS"
CHUNK_BYTES = 8 << 20  # 8 MiB

#: Packages worth recording when present. Absent ones are skipped silently --
#: the four conda environments each have a different subset.
TRACKED_DISTRIBUTIONS = [
    "chrombpnet",
    "duckdb",
    "finemo",
    "h5py",
    "hdf5plugin",
    "matplotlib",
    "modisco-lite",
    "MotifCompendium",
    "numpy",
    "pandas",
    "polars",
    "pyranges1",
    "pysam",
    "ruranges",
    "scipy",
    "tensorflow",
]


def repo_root() -> Path:
    """The repo root, located relative to this file."""
    return Path(__file__).resolve().parents[3]


def checksums_enabled() -> bool:
    return os.environ.get(CHECKSUM_ENV_VAR, "1").strip().lower() not in {"0", "false", "no"}


def md5sum(path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Streaming md5, identical to `md5sum(1)`. Constant memory."""
    digest = hashlib.md5()  # noqa: S324  # provenance fingerprint, not a security control
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def file_record(role: str, path, checksums: bool | None = None) -> dict:
    """Describe one input or output file. Never raises."""
    if checksums is None:
        checksums = checksums_enabled()
    p = Path(path)
    rec = {
        "role": role,
        "path": str(p.resolve()) if p.exists() else str(p),
        "exists": p.exists(),
        "size_bytes": None,
        "mtime": None,
        "md5": None,
        "md5_skipped": None,
    }
    if not p.exists() or not p.is_file():
        rec["md5_skipped"] = "missing" if not p.exists() else "not_a_file"
        return rec
    st = p.stat()
    rec["size_bytes"] = st.st_size
    rec["mtime"] = _utc(st.st_mtime)
    if not checksums:
        rec["md5_skipped"] = "disabled"
        return rec
    try:
        rec["md5"] = md5sum(p)
    except OSError as exc:
        rec["md5_skipped"] = f"error: {exc}"
    return rec


def normalize_remote_url(url: str | None) -> str | None:
    """Turn any GitHub remote form into a browsable https URL.

    Handles ``git@github.com:Owner/Repo.git``, ``ssh://git@github.com/Owner/Repo``
    and ``https://github.com/Owner/Repo.git``. Returns None for anything it does
    not recognise rather than guessing.
    """
    if not url:
        return None
    url = url.strip().removesuffix(".git")
    if url.startswith("git@"):  # git@host:owner/repo
        host, _, path = url[4:].partition(":")
        return f"https://{host}/{path}" if path else None
    if url.startswith("ssh://"):
        rest = url[len("ssh://") :]
        rest = rest.partition("@")[2] or rest
        return f"https://{rest}"
    if url.startswith(("https://", "http://")):
        return url
    return None


def git_info() -> dict:
    """Commit, branch, dirty flag and browsable GitHub URLs.

    All values are None if git or .git is unavailable (e.g. the repo was copied
    to the cluster as a tarball), which is why every probe is wrapped.

    ``commit_url`` and ``script_url`` are permalinks pinned to the commit that
    was checked out, so a record points at exactly the code that ran -- **unless
    ``dirty`` is true**, in which case the working tree had uncommitted changes
    and the link shows the committed version instead. Filter on ``git.dirty``
    before trusting a permalink for reproducibility.
    """

    def run(*args):
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    # `git remote get-url` arrived in git 2.7; Sherlock ships 1.8.3.1, where it
    # is an unknown subcommand and this returned None -- so commit_url and
    # script_url, the whole point of recording the remote, were silently absent
    # from every record. `git config --get` predates both and is what get-url
    # reads anyway.
    remote = run("config", "--get", "remote.origin.url")
    if remote is None:
        remote = run("remote", "get-url", "origin")
    repo_url = normalize_remote_url(remote)
    return {
        "commit": commit,
        "short_commit": commit[:8] if commit else None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "remote_url": repo_url,
        "commit_url": f"{repo_url}/commit/{commit}" if repo_url and commit else None,
    }


def script_url(script, commit: str | None, repo_url: str | None) -> str | None:
    """GitHub permalink to ``script`` at ``commit``, as a repo-relative blob URL."""
    if not (script and commit and repo_url):
        return None
    try:
        rel = Path(script).resolve().relative_to(repo_root())
    except (ValueError, OSError):
        return None
    return f"{repo_url}/blob/{commit}/{rel.as_posix()}"


def username() -> str | None:
    """The account that ran the step.

    ``$USER`` alone is unreliable: it is unset in a bare container, in some
    ``srun`` contexts, and under a cron-style launcher. ``getpass.getuser()``
    tries LOGNAME/USER/LNAME/USERNAME and then falls back to the password
    database, which is what actually works on a compute node.
    """
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        # No passwd entry for the uid -- happens in some container images.
        return os.environ.get("USER") or os.environ.get("LOGNAME")


def user_info() -> dict:
    """Who ran this, at both the account and the person level.

    ``user``/``uid`` identify the cluster account; ``git_user_name`` and
    ``git_user_email`` come from git config and identify the person, which is
    what you actually want when several people share a lab account or when an
    account name is opaque. Any of them may be None.
    """

    def git_config(key):
        try:
            out = subprocess.run(
                ["git", "config", "--get", key],
                cwd=repo_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() or None if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    try:
        uid = os.getuid()
    except AttributeError:  # non-POSIX
        uid = None

    return {
        "user": username(),
        "uid": uid,
        "git_user_name": git_config("user.name"),
        "git_user_email": git_config("user.email"),
    }


def slurm_info() -> dict:
    """Whatever SLURM put in the environment; all None outside a job."""

    def env(name):
        return os.environ.get(name) or None

    return {
        "job_id": env("SLURM_JOB_ID"),
        "array_job_id": env("SLURM_ARRAY_JOB_ID"),
        "array_task_id": env("SLURM_ARRAY_TASK_ID"),
        "job_name": env("SLURM_JOB_NAME"),
        "partition": env("SLURM_JOB_PARTITION"),
        "cpus": env("SLURM_CPUS_PER_TASK"),
        # Who submitted the job -- not always the process owner.
        "job_user": env("SLURM_JOB_USER"),
        "mem": env("SLURM_MEM_PER_NODE"),
        "nodelist": env("SLURM_JOB_NODELIST"),
    }


def tool_versions(extra: dict[str, str] | None = None) -> list[dict]:
    """Python plus whichever tracked distributions are installed."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    tools = [
        {"name": "python", "version": platform.python_version()},
    ]
    for name in TRACKED_DISTRIBUTIONS:
        try:
            tools.append({"name": name, "version": dist_version(name)})
        except PackageNotFoundError:
            continue
    for name, ver in (extra or {}).items():
        tools.append({"name": name, "version": str(ver)})
    return tools


class StepMetadata:
    """Accumulates one step's provenance record. Use via :func:`record`."""

    def __init__(self, step: str, dataset: str | None = None, out_dir=None, script=None):
        self.step = step
        self.dataset = dataset
        self.out_dir = Path(out_dir) if out_dir else None
        # The file implementing this step. Defaults to the running script; a
        # bash step passes its own path so the permalink points at the wrapper.
        self.script = (
            Path(script) if script else (Path(sys.argv[0]) if sys.argv and sys.argv[0] else None)
        )
        self.run_id = uuid.uuid4().hex
        self.started_monotonic = time.monotonic()
        self.started_at = _utc(time.time())
        self._inputs: list[tuple[str, Path]] = []
        self._outputs: list[tuple[str, Path]] = []
        self.params: list[dict] = []
        self.extra_tools: dict[str, str] = {}
        self.status = "ok"
        self.error: str | None = None
        self.exit_status = 0

    # -- declaration -----------------------------------------------------
    def add_input(self, role: str, path) -> None:
        self._inputs.append((role, Path(path)))

    def add_output(self, role: str, path) -> None:
        """Declare an output. It is hashed on exit, not now."""
        self._outputs.append((role, Path(path)))

    def add_param(self, key: str, value) -> None:
        self.params.append({"key": str(key), "value": "" if value is None else str(value)})

    def add_params(self, mapping) -> None:
        for k, v in dict(mapping).items():
            self.add_param(k, v)

    def add_tool(self, name: str, version) -> None:
        self.extra_tools[str(name)] = str(version)

    # -- emission --------------------------------------------------------
    def to_dict(self) -> dict:
        git = git_info()
        who = user_info()
        try:
            rel_script = (
                str(Path(self.script).resolve().relative_to(repo_root())) if self.script else None
            )
        except (ValueError, OSError):
            rel_script = str(self.script) if self.script else None
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "step": self.step,
            "dataset": self.dataset,
            "status": self.status,
            "exit_status": self.exit_status,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": _utc(time.time()),
            "duration_s": round(time.monotonic() - self.started_monotonic, 3),
            "host": socket.gethostname(),
            # Flat scalars: `GROUP BY "user"` is the common query, so these do
            # not get buried in a struct.
            "user": who["user"],
            "uid": who["uid"],
            "git_user_name": who["git_user_name"],
            "git_user_email": who["git_user_email"],
            "cwd": os.getcwd(),
            "command": list(sys.argv),
            "checksums_enabled": checksums_enabled(),
            "script": rel_script,
            "script_url": script_url(self.script, git.get("commit"), git.get("remote_url")),
            "git": git,
            "slurm": slurm_info(),
            "params": self.params,
            "inputs": [file_record(r, p) for r, p in self._inputs],
            "outputs": [file_record(r, p) for r, p in self._outputs],
            "tools": tool_versions(self.extra_tools),
        }

    def path(self) -> Path:
        stamp = self.started_at.replace(":", "").replace("-", "")
        return Path(self.out_dir) / self.step / f"{stamp}_{self.run_id[:8]}.json"

    def write(self, path=None) -> Path | None:
        """Write the record. Never raises -- provenance must not fail a step."""
        if self.out_dir is None and path is None:
            logger.debug("no metadata dir configured, not writing a record")
            return None
        target = Path(path) if path else self.path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        except OSError as exc:
            logger.warning("could not write run metadata to %s: %s", target, exc)
            return None
        logger.info("run metadata -> %s", target)
        return target


@contextmanager
def record(step: str, dataset: str | None = None, out_dir=None, script=None):
    """Context manager that writes a step's metadata on the way out.

    Writes on success *and* on failure, then re-raises. Outputs are hashed at
    exit, so declare them as soon as their paths are known.
    """
    md = StepMetadata(step, dataset=dataset, out_dir=out_dir, script=script)
    try:
        yield md
    except BaseException as exc:  # noqa: BLE001  # recorded, then re-raised
        md.status = "failed"
        md.error = f"{type(exc).__name__}: {exc}"
        md.exit_status = (
            exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        )
        if md.exit_status == 0:
            md.status = "ok"
            md.error = None
        md.write()
        raise
    else:
        md.write()


def default_dir(results_path=None) -> Path | None:
    """``<results_path>/metadata``, or ``$METADATA_DIR`` if set."""
    env = os.environ.get("METADATA_DIR")
    if env:
        return Path(env)
    if results_path:
        return Path(results_path) / "metadata"
    return None
