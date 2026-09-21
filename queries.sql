-- queries.sql — querying pipeline run metadata with DuckDB
--
-- Every step writes one JSON record per invocation to
-- <dataset>/results/metadata/<step>/<timestamp>_<runid>.json
-- (cross-dataset steps use <repo>/results/metadata/). They share one schema, so
-- every run of every step across every dataset loads as a single table.
--
--   duckdb -init queries.sql
--   -- or:  duckdb  then  .read queries.sql
--
-- Point RUNS at your collaboration root. union_by_name is required: steps carry
-- different params, and without it DuckDB would reject the mismatch.

CREATE OR REPLACE VIEW runs AS
SELECT * FROM read_json_auto(
    '/oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab/*/results/metadata/**/*.json',
    union_by_name = true
);

-- One row per file, inputs and outputs together: the provenance backbone.
CREATE OR REPLACE VIEW run_files AS
SELECT run_id, step, dataset, started_at, status, 'input' AS direction,
       f.role, f.path, f.md5, f.size_bytes, f.exists
FROM runs, UNNEST(inputs) AS t(f)
UNION ALL
SELECT run_id, step, dataset, started_at, status, 'output' AS direction,
       f.role, f.path, f.md5, f.size_bytes, f.exists
FROM runs, UNNEST(outputs) AS t(f);

-- Params in long form, so steps with different parameters coexist.
CREATE OR REPLACE VIEW run_params AS
SELECT run_id, step, dataset, p.key, p.value
FROM runs, UNNEST(params) AS t(p);


-- Step names have two granularities, deliberately:
--   '01.preprocess_peaks'  one sbatch job   (written by the EXIT trap)
--   'preprocess_peaks'     one tool call    (written by the Python script)
-- A job that loops over datasets yields one job record and several tool records.
CREATE OR REPLACE VIEW jobs  AS SELECT * FROM runs WHERE regexp_matches(step, '^[_0-9]');
CREATE OR REPLACE VIEW tools AS SELECT * FROM runs WHERE NOT regexp_matches(step, '^[_0-9]');


-- ─── what happened ───────────────────────────────────────────────────────────

-- Most recent run of every step, per dataset.
SELECT dataset, step, status, duration_s, started_at, git.short_commit
FROM runs QUALIFY row_number() OVER (PARTITION BY dataset, step ORDER BY started_at DESC) = 1
ORDER BY dataset, step;

-- Failures, newest first, with a link to the code that failed.
SELECT started_at, dataset, step, exit_status, error, script_url
FROM runs WHERE status <> 'ok' ORDER BY started_at DESC;

-- Steps that died before writing an output they had declared.
SELECT run_id, step, dataset, role, path
FROM run_files WHERE direction = 'output' AND NOT exists;

-- Where the wall-clock goes.
SELECT step, count(*) AS runs, round(avg(duration_s), 1) AS avg_s,
       round(max(duration_s), 1) AS max_s
FROM runs WHERE status = 'ok' GROUP BY step ORDER BY avg_s DESC;


-- Who ran what, and when. `user` is the cluster account; git_user_name is the
-- person, which is what you want when an account is shared or opaque.
SELECT "user", git_user_name, count(*) AS runs,
       count(DISTINCT dataset) AS datasets, max(started_at) AS last_run
FROM runs GROUP BY 1, 2 ORDER BY runs DESC;

-- Everything one person produced, newest first.
SELECT f.started_at, f.step, f.dataset, f.role, f.path
FROM run_files f JOIN runs r USING (run_id)
WHERE f.direction = 'output' AND r."user" = 'opushkar'
ORDER BY f.started_at DESC;


-- ─── provenance ──────────────────────────────────────────────────────────────

-- Which run produced this file, and from exactly which commit?
SELECT o.step, o.run_id, o.started_at, r.git.commit, r.script_url
FROM run_files o JOIN runs r USING (run_id)
WHERE o.direction = 'output' AND o.path LIKE '%chrombpnet_nobias.h5'
ORDER BY o.started_at DESC;

-- Build a dependency edge: an output of one run consumed as an input of another.
SELECT p.step AS produced_by, c.step AS consumed_by, p.path, p.md5
FROM run_files p JOIN run_files c ON p.md5 = c.md5 AND p.md5 IS NOT NULL
WHERE p.direction = 'output' AND c.direction = 'input' AND p.run_id <> c.run_id;

-- Did a file change between runs? Same path, different checksum.
SELECT path, count(DISTINCT md5) AS versions, min(started_at) AS first_seen,
       max(started_at) AS last_seen
FROM run_files WHERE md5 IS NOT NULL
GROUP BY path HAVING count(DISTINCT md5) > 1 ORDER BY versions DESC;

-- Runs from a dirty working tree: their script_url does NOT reflect what ran.
SELECT started_at, dataset, step, git.branch, git.short_commit
FROM runs WHERE git.dirty ORDER BY started_at DESC;


-- ─── reproducibility ─────────────────────────────────────────────────────────

-- Tool versions actually used, per step.
SELECT DISTINCT step, t.name, t.version
FROM runs, UNNEST(tools) AS u(t)
WHERE t.name IN ('chrombpnet', 'python', 'pyranges1', 'tensorflow', 'finemo')
ORDER BY step, t.name;

-- Did every dataset run step 01 with the same input window?
SELECT key, value, count(*) AS n, list(DISTINCT dataset) AS datasets
FROM run_params WHERE step = '01.preprocess_peaks' AND key = 'input_window'
GROUP BY key, value;

-- Export anything above as TSV (no second on-disk format needed):
--   COPY (SELECT * FROM run_files) TO 'run_files.tsv' (HEADER, DELIMITER '\t');
