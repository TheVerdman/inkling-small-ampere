"""Governed local phenomenon fixtures and deterministic prompt materialization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast

from inkling_ampere.manifests import manifest_digest
from inkling_ampere.mechanistic.contracts import (
    CapabilityState,
    MechanisticContractError,
    validate_probe_set,
)


def sealed_content_digest(value: Mapping[str, object]) -> str:
    """Digest a sealed document with its self-referential sealing field removed."""

    payload = copy.deepcopy(dict(value))
    payload.pop("sealing", None)
    return manifest_digest(payload)


def verify_probe_set_seal(value: Mapping[str, object]) -> str:
    validate_probe_set(value)
    sealing = value.get("sealing")
    if not isinstance(sealing, Mapping):
        raise MechanisticContractError("ProbeSet sealing record is invalid")
    observed = sealed_content_digest(value)
    expected = sealing.get("content_sha256")
    if observed != expected:
        raise MechanisticContractError(
            f"ProbeSet seal mismatch: expected {expected}, observed {observed}"
        )
    return observed


def materialize_probe(
    probe_set: Mapping[str, object], probe_id: str
) -> tuple[str, Mapping[str, object]]:
    """Materialize a public text fixture without a tokenizer or external sandbox."""

    verify_probe_set_seal(probe_set)
    capabilities = probe_set.get("modality_capabilities")
    if not isinstance(capabilities, Mapping):
        raise MechanisticContractError("ProbeSet capabilities are invalid")
    probes = probe_set.get("probes")
    if not isinstance(probes, list):
        raise MechanisticContractError("ProbeSet probes are invalid")
    selected: Mapping[str, object] | None = None
    for raw_probe in probes:
        if isinstance(raw_probe, Mapping) and raw_probe.get("id") == probe_id:
            selected = cast(Mapping[str, object], raw_probe)
            break
    if selected is None:
        raise MechanisticContractError(f"ProbeSet has no probe {probe_id!r}")
    modality = selected.get("modality")
    if modality != "text":
        raw_state = capabilities.get(str(modality))
        if raw_state != CapabilityState.GPU_VALIDATED.value:
            raise MechanisticContractError(
                f"{modality} fixture is gated until capability is gpu-validated"
            )
        raise MechanisticContractError(
            "media fixture materialization requires the later validated modality adapter"
        )
    generator = selected.get("generator")
    expected = selected.get("expected_outcome")
    if not isinstance(generator, Mapping) or not isinstance(expected, Mapping):
        raise MechanisticContractError("probe generator/expected outcome is invalid")
    kind = generator.get("kind")
    if kind == "literal":
        prompt = generator.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise MechanisticContractError("literal generator requires prompt")
        return prompt, cast(Mapping[str, object], expected)
    if kind == "needle-context":
        return _needle_context(generator), cast(Mapping[str, object], expected)
    if kind == "fact-integration":
        return _fact_integration(generator), cast(Mapping[str, object], expected)
    if kind == "tool-plan":
        return _tool_plan(generator), cast(Mapping[str, object], expected)
    raise MechanisticContractError(f"unsupported fixture generator {kind!r}")


def _needle_context(generator: Mapping[str, object]) -> str:
    filler = generator.get("filler")
    repetitions = generator.get("repetitions")
    needles = generator.get("needles")
    query = generator.get("query")
    if (
        not isinstance(filler, str)
        or not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
        or not isinstance(needles, list)
        or not isinstance(query, str)
    ):
        raise MechanisticContractError("needle-context generator fields are invalid")
    paragraphs = [f"[{index:05d}] {filler}" for index in range(repetitions)]
    for raw_needle in needles:
        if not isinstance(raw_needle, Mapping):
            raise MechanisticContractError("needle record must be an object")
        position = raw_needle.get("position")
        text = raw_needle.get("text")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not 0 <= position < repetitions
            or not isinstance(text, str)
        ):
            raise MechanisticContractError("needle position/text is invalid")
        paragraphs[position] = f"[{position:05d}] {text}"
    return "\n".join([*paragraphs, "", query])


def _fact_integration(generator: Mapping[str, object]) -> str:
    facts = generator.get("facts")
    question = generator.get("question")
    if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
        raise MechanisticContractError("fact-integration requires string facts")
    if not isinstance(question, str):
        raise MechanisticContractError("fact-integration requires a question")
    return "\n".join(
        ["Use all and only the following facts:", *[f"- {fact}" for fact in facts], "", question]
    )


def _tool_plan(generator: Mapping[str, object]) -> str:
    tools = generator.get("tools")
    state = generator.get("state")
    objective = generator.get("objective")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise MechanisticContractError("tool-plan requires string tool abstractions")
    if not isinstance(state, Mapping) or not isinstance(objective, str):
        raise MechanisticContractError("tool-plan state/objective is invalid")
    return (
        "This is an offline planning abstraction; do not execute tools.\n"
        f"Available tools: {', '.join(tools)}\n"
        f"State: {dict(state)}\n"
        f"Objective: {objective}\n"
        "Return a JSON array of tool names and include one recovery branch."
    )
