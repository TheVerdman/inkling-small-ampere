#!/usr/bin/env python3
"""Inspect all saved TP4 comparisons without rerunning GPUs or changing gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def generation_diagnostics(left, right):
    common_errors = []
    chosen_errors = []
    first_divergence = None
    shared_positions = 0
    for index, (x, y, token_x, token_y) in enumerate(
        zip(left["logprobs"], right["logprobs"], left["tokens"], right["tokens"], strict=False)
    ):
        shared_positions += 1
        common_errors.extend(abs(x[key] - y[key]) for key in x.keys() & y.keys())
        if str(token_x) in y:
            chosen_errors.append(abs(x[str(token_x)] - y[str(token_x)]))
        if token_x != token_y:
            first_divergence = {
                "zero_based_step": index,
                "triton_token": token_x,
                "flex_token": token_y,
                "triton_scores": {str(key): x.get(str(key)) for key in (token_x, token_y)},
                "flex_scores": {str(key): y.get(str(key)) for key in (token_x, token_y)},
            }
            break
    return {
        "exact_tokens_equal": left["tokens"] == right["tokens"],
        "triton_text": left["text"],
        "flex_text": right["text"],
        "shared_history_positions": shared_positions,
        "max_common_logprob_abs_difference": max(common_errors, default=None),
        "max_chosen_token_logprob_abs_difference": max(chosen_errors, default=None),
        "first_divergence": first_divergence,
    }


def layer_diagnostics(models):
    records = {
        name: {
            (worker["rank"], record["phase"], record["layer"]): record
            for worker in model["diagnostics"]["workers"]
            for record in worker["records"]
        }
        for name, model in models.items()
    }
    reference_summaries = {}
    for name, values in records.items():
        reference_summaries[name] = {}
        for metric in (
            "production_vs_fp32",
            "same_input_triton_vs_fp32",
            "same_input_triton_vs_flex",
            "bf16_rounding_vs_fp32",
        ):
            available = [
                (key, record[metric]) for key, record in values.items() if metric in record
            ]
            if not available:
                continue
            worst_key, worst = max(available, key=lambda item: item[1]["max_abs"])
            reference_summaries[name][metric] = {
                "calls": len(available),
                "max_abs": worst["max_abs"],
                "worst_call": {"rank": worst_key[0], "phase": worst_key[1], "layer": worst_key[2]},
                "maximum_relative_l2": max(value["relative_l2"] for _, value in available),
                "outside_0_02_plus_0_02_relative": sum(
                    value["outside_0_02_plus_0_02_relative"] for _, value in available
                ),
                "elements": sum(value["elements"] for _, value in available),
            }
    comparisons = {}
    for label, left, right in (
        ("flex_across_processes", "flex", "flex_repeat"),
        ("triton_vs_flex", "triton", "flex"),
    ):
        a, b = records[left], records[right]
        if a.keys() != b.keys():
            raise ValueError("layer observation coverage differs")
        rows = []
        omitted_long_decode = 0
        for key in sorted(a):
            x, y = a[key], b[key]
            if (x["q_len"], x["kv_len"]) != (y["q_len"], y["kv_len"]):
                raise ValueError("layer observation schedules differ")
            # The free first continuation token was not saved by the observer.
            # Its decode call is valid for same-input reference checks, but not
            # for claiming a matched history across independent processes.
            if key[1] == "long_decode":
                omitted_long_decode += 1
                continue
            row = {"rank": key[0], "phase": key[1], "layer": key[2], "fields": {}}
            for field in ("attention_input", "q", "rel", "attention_output"):
                s, t = x[field], y[field]
                if s["shape"] != t["shape"] or s["dtype"] != t["dtype"]:
                    raise ValueError("layer observation tensor geometry differs")
                errors = [abs(u - v) for u, v in zip(s["sample"], t["sample"], strict=True)]
                row["fields"][field] = {
                    "active_hash_equal": s["active_sha256"] == t["active_sha256"],
                    "last_token_hash_equal": s["last_token_sha256"] == t["last_token_sha256"],
                    "sample_max_abs_difference": max(errors),
                    "sample_rms_difference": math.sqrt(sum(v * v for v in errors) / len(errors)),
                }
            rows.append(row)
        short = [row for row in rows if row["phase"] == "short"]
        comparisons[label] = {
            "omitted_long_decode_records": omitted_long_decode,
            "long_decode_omission_reason": "free continuation token history was not recorded",
            "short_first_different_layer_by_rank": {
                str(rank): {
                    field: next(
                        (
                            row["layer"]
                            for row in short
                            if row["rank"] == rank and not row["fields"][field]["active_hash_equal"]
                        ),
                        None,
                    )
                    for field in ("attention_input", "q", "rel", "attention_output")
                }
                for rank in sorted({row["rank"] for row in short})
            },
            "records": rows,
        }
    observer_effect = {}
    for name, model in models.items():
        baseline = model["diagnostics"]["clean_repeats"][1]["fixed_history"]
        entries = []
        for observed in model["diagnostics"]["instrumented"]:
            errors, coverage_changed = [], False
            for a, b in zip(
                baseline[observed["probe"]]["logprobs"], observed["logprobs"], strict=True
            ):
                coverage_changed |= a.keys() != b.keys()
                errors.extend(abs(a[key] - b[key]) for key in a.keys() & b.keys())
            entries.append(
                {
                    "phase": observed["phase"],
                    "max_abs": max(errors),
                    "logprob_coverage_changed": coverage_changed,
                }
            )
        observer_effect[name] = entries
    return {
        "reference_errors": reference_summaries,
        "activation_comparisons": comparisons,
        "instrumented_vs_clean_prompt_scores": observer_effect,
    }


def analyze(report):
    left = report["production_models"]["triton"]
    right = report["production_models"]["flex"]
    rows = []
    all_errors = []
    histories_equal = True
    for probe, (a, b) in enumerate(zip(left["fixed_history"], right["fixed_history"], strict=True)):
        histories_equal &= a["prompt_tokens"] == b["prompt_tokens"] and a["suffix"] == b["suffix"]
        for position, (token, x, y) in enumerate(
            zip(a["suffix"], a["logprobs"], b["logprobs"], strict=True)
        ):
            if any(not math.isfinite(value) for value in list(x.values()) + list(y.values())):
                raise ValueError("nonfinite saved logprobs")
            common = x.keys() & y.keys()
            errors = [(abs(x[key] - y[key]), key) for key in common]
            all_errors.extend(error for error, _ in errors)
            best_x, best_y = max(x, key=x.get), max(y, key=y.get)
            worst, worst_token = max(errors)
            rows.append(
                {
                    "probe": probe,
                    "suffix_position": position,
                    "history_token_count": len(a["prompt_tokens"]) + position,
                    "shared_logprob_entries": len(common),
                    "mutual_top1_coverage": best_x in y and best_y in x,
                    "top1_equal": best_x == best_y,
                    "max_logprob_abs_difference": worst,
                    "worst_token": worst_token,
                    "worst_token_triton_logprob": x[worst_token],
                    "worst_token_flex_logprob": y[worst_token],
                    "actual_token_logprob_abs_difference": abs(x[str(token)] - y[str(token)]),
                }
            )
    smoke = [
        {"id": a["id"], **generation_diagnostics(a, b)}
        for a, b in zip(left["smoke"], right["smoke"], strict=True)
    ]
    source = report.get("source", {})
    result = {
        "recorded_run_status": report["status"],
        "recorded_error": report.get("error"),
        "offline_analysis_only": True,
        "matching_controls": {
            "runtime_config": left["runtime_config"] == right["runtime_config"],
            "tokenizer_preflight": left["tokenizer_preflight"] == right["tokenizer_preflight"],
            "parameter_sample_fingerprints": [
                (w["rank"], w["parameter_sample_sha256"]) for w in left["workers"]
            ]
            == [(w["rank"], w["parameter_sample_sha256"]) for w in right["workers"]],
            "fixed_histories": histories_equal,
            "fixed_schedule": left["fixed_schedule"] == right["fixed_schedule"],
        },
        "fixed_history": {
            "unchanged_absolute_tolerance": 0.1,
            "positions": len(rows),
            "shared_logprob_pairs": len(all_errors),
            "pairs_exceeding_tolerance": sum(error > 0.1 for error in all_errors),
            "max_logprob_abs_difference": max(all_errors),
            "max_actual_token_logprob_abs_difference": max(
                row["actual_token_logprob_abs_difference"] for row in rows
            ),
            "actual_token_positions_exceeding_tolerance": sum(
                row["actual_token_logprob_abs_difference"] > 0.1 for row in rows
            ),
            "top1_equal_positions": sum(row["top1_equal"] for row in rows),
            "minimum_shared_entries": min(row["shared_logprob_entries"] for row in rows),
            "coverage_failed_positions": sum(
                row["shared_logprob_entries"] < 28 or not row["mutual_top1_coverage"]
                for row in rows
            ),
            "positions_detail": rows,
        },
        "generation": {
            "exact_smoke_token_matches": sum(item["exact_tokens_equal"] for item in smoke),
            "smoke": smoke,
            "long": generation_diagnostics(left["long"], right["long"]),
            "proof": generation_diagnostics(left["proof"], right["proof"]),
        },
        "source_checks": {
            "before_matches": bool(source.get("expected_hashes"))
            and source.get("actual_hashes_before") == source.get("expected_hashes"),
            "after_check_recorded": "actual_hashes_after" in source,
            "after_matches": bool(source.get("expected_hashes"))
            and source.get("actual_hashes_after") == source.get("expected_hashes"),
            "compatibility_after_check_recorded": "compatibility_hashes_after" in report,
        },
    }
    if "diagnostic_summary" in report:
        result["diagnostic_summary"] = report["diagnostic_summary"]
        result["layer_diagnostics"] = layer_diagnostics(report["production_models"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.report.read_bytes()
    result = analyze(json.loads(content))
    result["source_report"] = str(args.report)
    result["source_report_sha256"] = hashlib.sha256(content).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result["fixed_history"].items() if k != "positions_detail"}))


if __name__ == "__main__":
    main()
