# Reproducible environment

## Status

The repository bootstrap is complete. Historical physical/topology evidence
and a successful live run of the pinned vLLM image are both preserved. The
live run confirms four A100s, CUDA/NCCL operation, W8A16 scheme construction,
and the source-predicted Inkling attention failure.

## Pinned environments

The lightweight developer environment is locked by `pyproject.toml`,
`.python-version`, and `uv.lock`:

| Component | Pin |
| --- | ---: |
| Python | 3.12.12 |
| uv | 0.11.29 |
| Hatchling | 1.31.0 |
| pytest | 9.1.1 |
| Ruff | 0.15.22 |
| mypy | 2.3.0 |

The serving environment starts from:

```text
vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404
sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8
```

The image metadata records CUDA 12.9.1, Python 3.12, A100 `sm80` build support,
and vLLM commit `ffd46bfab2128bb84146050e98b51a617c6575ab`. The full digest pins
the transitive CUDA, NCCL, attention-kernel, and Python environment even where
an individual version is not independently selected.

Sources:

- [Pinned vLLM image metadata](https://hub.docker.com/layers/vllm/vllm-openai/v0.26.0-x86_64-cu129-ubuntu2404/images/sha256-4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8)
- [vLLM 0.26.0 package metadata](https://pypi.org/project/vllm/0.26.0/)
- [Complete checked-in software lock](../configs/hardware/software-lock.json)

## Developer setup

`bootstrap.sh` installs the exact uv version into the ignored `.tools`
directory, then synchronizes the frozen lock:

```bash
make bootstrap
. .venv/bin/activate
make check
```

No CUDA package is installed in the developer environment. Checkpoint
inspection and the future memory model should remain runnable without a GPU.

## Observed four-A100 hardware

The successful Vertex job `8664974044392587264` ran on
`a2-ultragpu-4g` with four `NVIDIA_A100_80GB` accelerators on 2026-06-20.
Its preserved artifacts establish:

- four A100-SXM4-80GB devices at compute capability 8.0;
- 81,920 MiB physical HBM per device, with MIG disabled;
- peer access between every pair;
- `NV12` between every distinct GPU pair and 12 reported 25 GB/s links per
  device; and
- a correct four-rank NCCL all-reduce with zero maximum absolute error.

The checked local copies and SHA-256 values are recorded in
`manifests/hardware-reference.json`. The source driver was 535.288.01; that
value is historical evidence, not a claim about the pinned vLLM image run.

## Live pinned-image result

Vertex job `7887664704179404800` succeeded on 2026-07-30 using the exact
serving-image manifest. It observed:

- four `NVIDIA A100-SXM4-80GB` devices at compute capability 8.0;
- 84,987,740,160 usable bytes/device from `torch.cuda`;
- PyTorch 2.11.0+cu129, CUDA 12.9, and NCCL 2.28.9;
- a correct four-rank NCCL all-reduce with sum 10.0 on every rank;
- successful construction of group-size-128 symmetric W8A16; and
- `NotImplementedError` from Inkling relative attention because custom
  `score_mod` is unsupported on SM8x.

The image does not include `nvidia-smi`, so the live report does not supersede
the historical topology/driver evidence. The complete artifact and its
checksum are pinned in `manifests/a100-recon-20260730.json`.

## Environment inventory

Run:

```bash
make doctor
```

The command captures:

- OS, kernel, architecture, CPU count, host RAM, and Python.
- NVIDIA driver, GPU identity, total and free HBM.
- `nvidia-smi -q` and the NVLink/PCIe topology matrix.
- CUDA compiler and PyTorch CUDA runtime information.
- NUMA and block-device inventories.
- Storage capacity and a small, fsync-backed sequential diagnostic probe.
- Installed serving-package versions.
- Project commit and dirty-state metadata.

Missing platform-specific tools are recorded as unavailable rather than causing
collection to fail. Every report is canonical JSON at
`results/raw/env-<manifest-hash>.json`, created with no-overwrite semantics.

The storage probe is intentionally small and is not suitable for publication
benchmarks. Disable it with:

```bash
make doctor DOCTOR_ARGS="--storage-benchmark-mib 0"
```

## Target-node procedure

1. Mount or identify the 1.5 TB Local SSD.
2. Run a host report:

   ```bash
   make doctor STORAGE_PATH=/path/to/local-ssd
   ```

3. Build the pinned image:

   ```bash
   docker build --tag inkling-small-ampere:bootstrap .
   ```

4. Run strict validation inside the container with all GPUs and the Local SSD
   visible:

   ```bash
   docker run --rm --gpus all --ipc=host \
     --volume "$PWD:/workspace/inkling-small-ampere" \
     --volume /path/to/local-ssd:/local-ssd \
     --workdir /workspace/inkling-small-ampere \
     inkling-small-ampere:bootstrap \
     make doctor-strict STORAGE_PATH=/local-ssd
   ```

Strict mode exits with status 2 unless every hardware and serving-software
check passes. A non-target developer report can still be collected successfully;
its validation section will say `ready: false`.

For a bounded Vertex hardware/runtime probe without model weights, run:

```bash
bash scripts/gcp/submit_vertex_recon.sh
```

It requests exactly one `a2-ultragpu-4g` worker, disables retries, limits
container execution to 30 minutes, exercises four-rank NCCL, constructs the
pinned W8A16 scheme, calls Inkling relative attention once, and uploads a
structured JSON artifact. Vertex queue time is not covered by the execution
timeout and can be longer during regional A100 contention.

## Report handling

- Do not edit or overwrite a generated report.
- Preserve the full JSON and record its `run_id` in experiment manifests.
- If hardware, driver, container, mount, or package state changes, collect a
  new report.
- Record sustained Local SSD throughput separately during benchmark setup.
- Never treat `nvidia-smi`'s maximum supported CUDA version as the installed
  CUDA runtime version.
