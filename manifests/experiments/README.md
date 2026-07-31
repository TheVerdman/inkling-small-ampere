# Experiment manifests

Create one immutable JSON manifest per run. Populate exact 40-character source,
runtime, project, and dataset revisions; derive the run ID with
`inkling_ampere.manifests.manifest_run_id`; never overwrite an existing run.

