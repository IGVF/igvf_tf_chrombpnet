-- queries.sql — querying pipeline run metadata with DuckDB
--
-- Every step writes one JSON record per invocation to
-- <dataset>/results/metadata/<timestamp>_<step>_<runid>.json
-- (cross-dataset steps use <repo>/results/metadata/). The directory is FLAT --
-- the step is in the filename, not a parent directory -- because `step` is a
-- column and the filtering belongs here, in SQL, not in the directory tree.
-- They share one schema, so every run of every step across every dataset loads
-- as a single table.
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

-- ONE long-format table: every input, output, parameter and metric a run
-- recorded, one row each. parameter_type tells the four kinds apart, so a
-- single UNNEST answers "what did this run consume, produce, and with what
-- settings". Every column is meaningful on its own -- there is deliberately no
-- bare `key`, `value`, `name` or `path`.
CREATE OR REPLACE VIEW run_parameters AS
SELECT uuid, step, dataset, date_created, run_status,
       p.parameter_name, p.parameter_value, p.parameter_type,
       p.file, p.filepath, p.file_format,
       p.file_exists, p.file_size, p.file_mtime, p.md5sum, p.md5_skipped
FROM runs, UNNEST(parameters) AS t(p);

-- Just the files. parameter_name says WHAT the data is (signal, fragments,
-- peaks, contributions); file_format says how it is encoded (bigwig, tsv.gz,
-- narrowPeak, h5). The two never borrow each other's vocabulary, so
-- "every representation of `contributions`" is one WHERE clause.
CREATE OR REPLACE VIEW run_files AS
SELECT * FROM run_parameters WHERE parameter_type IN ('input', 'output');

-- Settings and measurements, which are NOT the same thing: `param` controlled
-- the run, `metric` was produced by it.
CREATE OR REPLACE VIEW run_settings AS
SELECT uuid, step, dataset, parameter_name, parameter_value
FROM run_parameters WHERE parameter_type = 'param';

CREATE OR REPLACE VIEW run_metrics AS
SELECT uuid, step, dataset, parameter_name, parameter_value
FROM run_parameters WHERE parameter_type = 'metric';


-- Step names have two granularities, deliberately:
--   '00.1.preprocess_peaks'  one sbatch job   (written by the EXIT trap)
--   'preprocess_peaks'       one tool call    (written by the Python script)
-- A job that loops over datasets yields one job record and several tool records.
-- The leading digit is the ONLY discriminator now that the layout is flat, which
-- is what these two views exist to hide. To pair a job with the tool calls it
-- ran, join on time containment rather than on the name:
--   FROM jobs j JOIN tools t
--     ON t.date_created >= j.date_created AND t.date_completed <= j.date_completed
CREATE OR REPLACE VIEW jobs  AS SELECT * FROM runs WHERE regexp_matches(step, '^[_0-9]');
CREATE OR REPLACE VIEW tools AS SELECT * FROM runs WHERE NOT regexp_matches(step, '^[_0-9]');


-- ─── what happened ───────────────────────────────────────────────────────────

-- Most recent run of every step, per dataset.
SELECT dataset, step, run_status, duration_s, date_created, git.short_commit
FROM runs QUALIFY row_number() OVER (PARTITION BY dataset, step ORDER BY date_created DESC) = 1
ORDER BY dataset, step;

-- Failures, newest first, with a link to the code that failed.
SELECT date_created, dataset, step, exit_status, error, script_url
FROM runs WHERE run_status <> 'ok' ORDER BY date_created DESC;

-- Steps that died before writing an output they had declared.
SELECT uuid, step, dataset, parameter_name, filepath
FROM run_files WHERE parameter_type = 'output' AND NOT file_exists;

-- Where the wall-clock goes.
SELECT step, count(*) AS runs, round(avg(duration_s), 1) AS avg_s,
       round(max(duration_s), 1) AS max_s
FROM runs WHERE run_status = 'ok' GROUP BY step ORDER BY avg_s DESC;


-- Who ran what, and when. `user` is the cluster account; git_user_name is the
-- person, which is what you want when an account is shared or opaque.
SELECT "user", git_user_name, count(*) AS runs,
       count(DISTINCT dataset) AS datasets, max(date_created) AS last_run
FROM runs GROUP BY 1, 2 ORDER BY runs DESC;

-- Everything one person produced, newest first.
SELECT f.date_created, f.step, f.dataset, f.parameter_name, f.file
FROM run_files f JOIN runs r USING (uuid)
WHERE f.parameter_type = 'output' AND r."user" = 'opushkar'
ORDER BY f.date_created DESC;


-- ─── provenance ──────────────────────────────────────────────────────────────

-- Which run produced this file, and from exactly which commit?
SELECT o.step, o.uuid, o.date_created, r.git.commit, r.script_url
FROM run_files o JOIN runs r USING (uuid)
WHERE o.parameter_type = 'output' AND o.file = 'chrombpnet_nobias.h5'
ORDER BY o.date_created DESC;

-- Build a dependency edge: an output of one run consumed as an input of another.
SELECT p.step AS produced_by, c.step AS consumed_by, p.filepath, p.md5sum
FROM run_files p JOIN run_files c ON p.md5sum = c.md5sum AND p.md5sum IS NOT NULL
WHERE p.parameter_type = 'output' AND c.parameter_type = 'input'
  AND p.uuid <> c.uuid;

-- Did a file change between runs? Same path, different checksum.
SELECT filepath, count(DISTINCT md5sum) AS versions, min(date_created) AS first_seen,
       max(date_created) AS last_seen
FROM run_files WHERE md5sum IS NOT NULL
GROUP BY filepath HAVING count(DISTINCT md5sum) > 1 ORDER BY versions DESC;

-- Runs from a dirty working tree: their script_url does NOT reflect what ran.
SELECT date_created, dataset, step, git.branch, git.short_commit
FROM runs WHERE git.dirty ORDER BY date_created DESC;


-- ─── reproducibility ─────────────────────────────────────────────────────────

-- Tool versions actually used, per step.
SELECT DISTINCT step, t.software_name, t.software_version
FROM runs, UNNEST(software_versions) AS u(t)
WHERE t.software_name IN ('chrombpnet', 'python', 'pyranges1', 'tensorflow', 'finemo')
ORDER BY step, t.software_name;

-- Did every dataset run step 01 with the same input window?
SELECT parameter_name, parameter_value, count(*) AS n, list(DISTINCT dataset) AS datasets
FROM run_settings WHERE step = '00.1.preprocess_peaks' AND parameter_name = 'input_window'
GROUP BY key, value;


-- ─── dataset triage, before spending GPU time ────────────────────────────────

-- Rank datasets by how well signal separates peaks from their GC-matched
-- background (02.0). This is the task ChromBPNet is trained on, so a value
-- near 0.5 means the dataset has nothing to teach it. Read alongside
-- tss_enrichment: low on both usually means the signal and the peaks came from
-- different samples.
WITH qc AS (
    SELECT dataset, parameter_name AS k, TRY_CAST(parameter_value AS DOUBLE) AS v
    FROM run_metrics WHERE step = 'qc_signal'
)
SELECT dataset,
       max(v) FILTER (k = 'auroc_peaks_vs_nonpeaks')             AS auroc,
       max(v) FILTER (k = 'signal_enrichment_peak_over_nonpeak') AS enrichment,
       max(v) FILTER (k = 'tss_enrichment')                      AS tsse,
       max(v) FILTER (k = 'frac_peaks_zero_signal')              AS frac_empty_peaks,
       max(v) FILTER (k = 'total_insertions')                    AS insertions
FROM qc GROUP BY dataset ORDER BY auroc;

-- The negatives that a given model was trained on are reproducible only from
-- the seed AND the ChromBPNet version that consumed it.
SELECT r.dataset, r.date_created, p.parameter_value AS seed,
       t.software_version AS chrombpnet
FROM runs r
JOIN run_settings p ON p.uuid = r.uuid AND p.parameter_name = 'seed'
LEFT JOIN (SELECT uuid, t.software_version FROM runs, UNNEST(software_versions) AS u(t)
           WHERE t.software_name = 'chrombpnet') t ON t.uuid = r.uuid
WHERE r.step = '01.0.preprocess_nonpeaks' ORDER BY r.date_created DESC;


-- Export anything above as TSV (no second on-disk format needed):
--   COPY (SELECT * FROM run_files) TO 'run_files.tsv' (HEADER, DELIMITER '\t');
