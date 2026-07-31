# Benchmark methodology

Status: **protocol skeleton; no measurements**.

Every published result will record warm-up, repetitions, median, P10/P90,
hardware state, software and model revisions, serving flags, context, batch
size, and quantization profile. Cold load, warm startup, time to first token,
prefill, decode, end-to-end latency, HBM, host RAM, interconnect traffic, and
power will be separated.

The small storage probe in `make doctor` is diagnostic only and must not be
presented as sustained Local SSD performance.

