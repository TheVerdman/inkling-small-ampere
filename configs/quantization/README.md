# Quantization profiles

The three Work Order 002 projections use symmetric groupwise W8A16 with
BF16 scales and no zero points:

- `w8a16-conservative-v1.json` follows the official NVFP4 sensitivity prior.
- `w8a16-balanced-v1.json` adds large attention and dense-MLP matrices.
- `w8a16-fit-first-v1.json` is an explicitly runtime-incompatible lower bound.

The profiles operate on the verified safetensors inventory. MTP is inventoried
but omitted from the first text-only proof of life because vLLM treats it as an
optional speculative-decoding checkpoint.
