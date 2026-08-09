# Inkling-Small header inventory fixture

`tensor_inventory.csv` is an exact byte-for-byte copy of the header-only
inventory generated for `thinkingmachines/Inkling-Small` revision
`b2d4f225a02032c5d154bff748ab5a00c5ca26e4`. It contains tensor names, shapes,
dtypes, shard assignments, byte ranges, and classification metadata; it
contains no tensor payloads or model weights.

- Records: 1,048 plus the CSV header
- Size: 277,473 bytes
- SHA-256: `7882601238779edb8ed80d39f1fbe6adc2cb6f2f2a4941633ac4186b3ccd0355`
- Source manifest: `manifests/source-checkpoint.json`
- Conversion plan: `conversion-e747e8121d5cd12c54c9`

The same inventory hash is pinned by the successful Gate D run manifest for
the immutable source revision. This compact metadata fixture is deliberately
tracked so `make check` reproduces the exact full conversion-plan validation
from a clean checkout. Generated operational evidence under `results/` remains
ignored and is not a test prerequisite.
