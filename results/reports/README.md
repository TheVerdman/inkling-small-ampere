# Generated reports

Generated reports live here. Commit a report only after its underlying immutable
manifests and data provenance are available.

This ignored directory is operational evidence, not a test-fixture location.
The exact header-only inventory needed by the hermetic conversion-plan test is
tracked at
`tests/fixtures/inkling-small-b2d4f225/tensor_inventory.csv`, with its source
revision and SHA-256 documented beside it. A clean `make check` must not depend
on preserved files under `results/`.
