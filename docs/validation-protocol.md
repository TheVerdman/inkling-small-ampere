# Validation protocol

Status: **protocol skeleton**.

Validation proceeds from structural integrity to module numerics, router
stability, logit agreement, and task-level behavior. Exact requests, assets,
hashes, seeds, effort settings, provider revisions, outputs, and finish metadata
must be retained.

Initial release targets:

- No unexplained category-level loss greater than three percentage points.
- At least 98% weighted aggregate task retention against the chosen reference.
- No material structured-output, executable-code, multimodal, safety, or
  instruction-following regression.
- Router top-k agreement reported by layer and modality.

These are investigation thresholds, not evidence that the model currently
passes.

