"""Pin the motif databases utils.references fetches.

Both are plain files on GitHub, with no metadata API to consult, so the md5
pinned in references.py is the whole integrity check. These tests make sure
the URLs cannot float and that the check runs on every fetch, including over a
file that is already present. Nothing here touches the network: ``download``
is replaced by a stub that writes known bytes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

from utils import references  # noqa: E402

MOTIFCOMPENDIUM_V1_0_19 = "7e9d1c2f932ea7f53fd36e174affef26eaf086c4"


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324


# ── the URLs are pinned ──────────────────────────────────────────────────────


def test_motifcompendium_db_url_is_pinned_to_a_commit_not_main():
    """The file changed at v1.0.17; a `main` URL gives different installs
    different annotation databases without anyone noticing."""
    url = references.layout("/tmp/refs")["ref_db_meme_url"]
    assert "/main/" not in url
    assert f"/{MOTIFCOMPENDIUM_V1_0_19}/" in url
    assert url.endswith("/pipeline/data/MotifCompendium-Database-Human.meme.txt")


def test_motifcompendium_db_md5_is_pinned_and_exported():
    lay = references.layout("/tmp/refs")
    assert lay["ref_db_meme_md5"] == "b54441e0bfb9623345b6802763472d3c"
    assert "ref_db_meme_md5=b54441e0bfb9623345b6802763472d3c" in references.to_shell(lay)


def test_motifcompendium_db_file_name_carries_the_version():
    """A copy fetched from `main` before the pin keeps the unversioned name, so
    the pinned file is downloaded beside it rather than the old one reused."""
    path = Path(references.layout("/tmp/refs")["ref_db_meme"])
    assert path.name == f"MotifCompendium-{references.MOTIF_DB_VERSION}-Database-Human.meme.txt"
    assert path.parent == Path("/tmp/refs/motif")


def test_chrombpnet_motif_db_is_pinned_to_a_tag_and_md5():
    lay = references.layout("/tmp/refs")
    assert "/v1.0.1/" in lay["chrombpnet_motifs_url"]
    assert lay["chrombpnet_motifs_md5"] == "30d5f17c169eb2ea0588c7546acfbf4d"


# ── the pinned md5 is checked ────────────────────────────────────────────────


def test_check_pinned_md5_rejects_a_mismatch(tmp_path):
    f = tmp_path / "db.meme.txt"
    f.write_bytes(b"MEME version 4\n")
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        references._check_pinned_md5(f, "0" * 32, "thing")


def test_check_pinned_md5_accepts_a_match(tmp_path):
    f = tmp_path / "db.meme.txt"
    f.write_bytes(b"MEME version 4\n")
    references._check_pinned_md5(f, _md5(b"MEME version 4\n"), "thing")


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the network fetch; record the URLs asked for."""
    calls = []

    def fake(url, dest, retries=3, log=print):
        calls.append(url)
        Path(dest).write_bytes(fake.payload)
        return Path(dest)

    fake.payload = b"MEME version 4\n"
    monkeypatch.setattr(references, "download", fake)
    return fake, calls


def test_fetch_pinned_downloads_an_absent_file_and_checks_it(tmp_path, fake_download):
    fake, calls = fake_download
    dest = tmp_path / "db.meme.txt"
    references._fetch_pinned(
        "https://example.invalid/db", dest, _md5(fake.payload), "db", log=lambda *_: None
    )
    assert calls == ["https://example.invalid/db"]
    assert dest.read_bytes() == fake.payload


def test_fetch_pinned_rejects_a_download_with_the_wrong_md5(tmp_path, fake_download):
    fake, _ = fake_download
    fake.payload = b"not the pinned file\n"
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        references._fetch_pinned(
            "https://example.invalid/db",
            tmp_path / "db.meme.txt",
            _md5(b"MEME version 4\n"),
            "db",
            log=lambda *_: None,
        )


def test_fetch_pinned_checks_a_file_that_is_already_present(tmp_path, fake_download):
    """Present files are not re-downloaded, but they are still verified: that is
    what catches a copy made before the pin."""
    _, calls = fake_download
    dest = tmp_path / "db.meme.txt"
    dest.write_bytes(b"an older revision\n")
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        references._fetch_pinned(
            "https://example.invalid/db",
            dest,
            _md5(b"MEME version 4\n"),
            "db",
            log=lambda *_: None,
        )
    assert calls == []


def test_fetch_pinned_leaves_a_verified_present_file_alone(tmp_path, fake_download):
    fake, calls = fake_download
    dest = tmp_path / "db.meme.txt"
    dest.write_bytes(fake.payload)
    references._fetch_pinned(
        "https://example.invalid/db", dest, _md5(fake.payload), "db", log=lambda *_: None
    )
    assert calls == []
