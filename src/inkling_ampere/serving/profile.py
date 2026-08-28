"""Strict, dependency-free loading for Inkling serving profiles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ServingProfileError(ValueError):
    """Raised when a serving profile is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ModelSettings:
    served_model_name: str
    checkpoint_id: str
    quantization: str
    artifact_uri: str
    conversion_manifest_sha256: str


@dataclass(frozen=True)
class ApiSettings:
    primary_protocol: str
    models_route: str
    responses_route: str
    capabilities_route: str
    streaming_terminal_event: str
    structured_json_schema: bool
    response_store_enabled: bool
    chat_completions_contract: bool


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int


@dataclass(frozen=True)
class RuntimeSettings:
    vllm_version: str
    vllm_commit: str
    tensor_parallel_size: int
    expert_parallel_size: int
    dtype: str
    tokenizer_mode: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int
    kv_cache_memory_bytes: int
    cpu_offload_gib: float
    enforce_eager: bool
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    async_scheduling: bool
    language_model_only: bool
    disable_custom_all_reduce: bool
    distributed_executor_backend: str
    reasoning_parser: str
    tool_call_parser: str
    enable_auto_tool_choice: bool
    performance_mode: str
    marlin_use_atomic_add: bool
    seed: int


@dataclass(frozen=True)
class ValidationSettings:
    model_runtime: str
    responses_api: str
    long_context: str
    maximum_verified_model_len: int


@dataclass(frozen=True)
class MemoryProjection:
    estimated_required_kv_cache_bytes: int
    allocated_kv_cache_bytes: int
    evidence: str


@dataclass(frozen=True)
class RuntimePatch:
    path: str
    sha256: str


@dataclass(frozen=True)
class ManifestPin:
    path: str
    sha256: str


@dataclass(frozen=True)
class ProcessorSettings:
    assets: tuple[ManifestPin, ...]
    image_token_id: int
    audio_token_id: int
    image_placeholder: str
    audio_placeholder: str
    required_weight_prefixes: tuple[str, ...]
    image_patch_size: int
    image_rescale_factor: float
    image_rescale_max_upscaled_long_edge: int
    audio_token_duration_seconds: float
    audio_samples_per_token: int


@dataclass(frozen=True)
class ModalityValidation:
    status: str
    native_processor: str
    native_engine: str
    responses_api: str
    context_ladder: str
    maximum_verified_context_tokens: int


@dataclass(frozen=True)
class ImageInputSettings:
    enabled: bool
    formats: tuple[str, ...]
    mime_types: tuple[str, ...]
    transports: tuple[str, ...]
    maximum_items: int
    maximum_base64_characters: int
    maximum_file_bytes: int
    maximum_decoded_bytes: int
    maximum_width: int
    maximum_height: int
    maximum_pixels: int
    maximum_decompression_ratio: float
    maximum_processor_tokens: int
    validation: ModalityValidation


@dataclass(frozen=True)
class AudioInputSettings:
    enabled: bool
    formats: tuple[str, ...]
    mime_types: tuple[str, ...]
    transports: tuple[str, ...]
    maximum_items: int
    maximum_base64_characters: int
    maximum_file_bytes: int
    maximum_decoded_bytes: int
    maximum_duration_seconds: float
    maximum_frames: int
    sample_rates_hz: tuple[int, ...]
    channels: tuple[int, ...]
    sample_width_bytes: tuple[int, ...]
    maximum_processor_tokens: int
    validation: ModalityValidation


@dataclass(frozen=True)
class MixedMediaSettings:
    enabled: bool
    maximum_total_items: int
    maximum_total_decoded_bytes: int
    maximum_processor_tokens: int
    validation: ModalityValidation


@dataclass(frozen=True)
class PromotionEvidence:
    attestation_sha256: str
    attestation_payload_sha256: str
    candidate_profile_sha256: str
    serving_image_uri: str
    responses_edge_image_uri: str
    responses_edge_revision: str


@dataclass(frozen=True)
class MultimodalSettings:
    enabled: bool
    scope: str
    output_modalities: tuple[str, ...]
    maximum_request_bytes: int
    maximum_context_tokens: int
    minimum_text_context_reserve_tokens: int
    research_manifest: ManifestPin | None
    processor: ProcessorSettings | None
    image: ImageInputSettings
    audio: AudioInputSettings
    mixed_media: MixedMediaSettings
    promotion_evidence: PromotionEvidence | None


@dataclass(frozen=True)
class ServingProfile:
    """Validated launch and capability configuration for one service shape."""

    path: Path
    profile_sha256: str
    schema_version: str
    kind: str
    profile_id: str
    status: str
    description: str
    model: ModelSettings
    api: ApiSettings
    server: ServerSettings
    runtime: RuntimeSettings
    validation: ValidationSettings
    memory_projection: MemoryProjection
    multimodal: MultimodalSettings
    patches: tuple[RuntimePatch, ...]

    def multimodal_limit_per_prompt(self) -> dict[str, int | dict[str, int]]:
        """Return vLLM item and profiling bounds from the admission contract."""

        if not self.multimodal.enabled:
            return {}
        return {
            "audio": {
                "count": self.multimodal.audio.maximum_items,
                "length": self.multimodal.audio.maximum_frames,
            },
            "image": {
                "count": self.multimodal.image.maximum_items,
                "height": self.multimodal.image.maximum_height,
                "width": self.multimodal.image.maximum_width,
            },
        }

    def vllm_command(
        self,
        model_path: Path,
        *,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """Build the exact pinned vLLM command for this profile."""

        runtime = self.runtime
        command = [
            "vllm",
            "serve",
            str(model_path),
            "--served-model-name",
            self.model.served_model_name,
            "--host",
            host if host is not None else self.server.host,
            "--port",
            str(port if port is not None else self.server.port),
            "--tensor-parallel-size",
            str(runtime.tensor_parallel_size),
            "--distributed-executor-backend",
            runtime.distributed_executor_backend,
            "--dtype",
            runtime.dtype,
            "--tokenizer-mode",
            runtime.tokenizer_mode,
            "--max-model-len",
            str(runtime.max_model_len),
            "--max-num-seqs",
            str(runtime.max_num_seqs),
            "--max-num-batched-tokens",
            str(runtime.max_num_batched_tokens),
            "--block-size",
            str(runtime.block_size),
            "--kv-cache-memory-bytes",
            str(runtime.kv_cache_memory_bytes),
            "--cpu-offload-gb",
            str(runtime.cpu_offload_gib),
            "--reasoning-parser",
            runtime.reasoning_parser,
            "--tool-call-parser",
            runtime.tool_call_parser,
            "--middleware",
            "inkling_ampere.serving.middleware.PadawanCapabilitiesMiddleware",
            "--seed",
            str(runtime.seed),
            "--generation-config",
            "vllm",
            "--enable-request-id-headers",
            "--disable-fastapi-docs",
        ]
        command.append(
            "--enable-chunked-prefill"
            if runtime.enable_chunked_prefill
            else "--no-enable-chunked-prefill"
        )
        command.append(
            "--enable-prefix-caching"
            if runtime.enable_prefix_caching
            else "--no-enable-prefix-caching"
        )
        command.append(
            "--async-scheduling" if runtime.async_scheduling else "--no-async-scheduling"
        )
        if runtime.enforce_eager:
            command.append("--enforce-eager")
        if runtime.language_model_only:
            command.append("--language-model-only")
        if self.multimodal.enabled:
            command.extend(
                [
                    "--limit-mm-per-prompt",
                    json.dumps(
                        self.multimodal_limit_per_prompt(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        if runtime.disable_custom_all_reduce:
            command.append("--disable-custom-all-reduce")
        if runtime.enable_auto_tool_choice:
            command.append("--enable-auto-tool-choice")
        if api_key:
            command.extend(["--api-key", api_key])
        return command

    def capability_document(self) -> dict[str, Any]:
        """Describe supported and projected behavior without overstating validation."""

        continuation_supported = self.api.response_store_enabled
        return {
            "schema_version": "1.1.0",
            "service": "inkling-small-ampere",
            "profile_id": self.profile_id,
            "profile_status": self.status,
            "profile_sha256": self.profile_sha256,
            "model": {
                "id": self.model.served_model_name,
                "checkpoint_id": self.model.checkpoint_id,
                "quantization": self.model.quantization,
            },
            "protocol": {
                "primary": self.api.primary_protocol,
                "routes": {
                    "models": self.api.models_route,
                    "responses": self.api.responses_route,
                    "capabilities": self.api.capabilities_route,
                },
                "chat_completions_contract": self.api.chat_completions_contract,
            },
            "features": {
                "streaming": {
                    "supported": True,
                    "terminal_event": self.api.streaming_terminal_event,
                    "validation": self.validation.responses_api,
                },
                "structured_outputs": {
                    "json_schema": self.api.structured_json_schema,
                    "validation": self.validation.responses_api,
                },
                "function_tools": {
                    "supported_by_runtime": True,
                    "parser": self.runtime.tool_call_parser,
                    "validation": self.validation.responses_api,
                },
                "reasoning": {
                    "parser": self.runtime.reasoning_parser,
                    "validation": self.validation.responses_api,
                },
                "previous_response_id": {
                    "supported": continuation_supported,
                    "requires_store_true": True,
                    "scope": "single-process-memory" if continuation_supported else "disabled",
                    "durable": False,
                    "replica_safe": False,
                    "reason": (
                        "vLLM storage is process-local, unbounded, and intentionally disabled"
                        if not continuation_supported
                        else "enabled only for a single-replica bounded experiment"
                    ),
                },
            },
            "modalities": self._modality_capabilities(),
            "runtime": {
                "vllm_version": self.runtime.vllm_version,
                "vllm_commit": self.runtime.vllm_commit,
                "performance_mode": self.runtime.performance_mode,
                "tensor_parallel_size": self.runtime.tensor_parallel_size,
                "dtype": self.runtime.dtype,
                "max_model_len": self.runtime.max_model_len,
                "max_num_seqs": self.runtime.max_num_seqs,
                "max_num_batched_tokens": self.runtime.max_num_batched_tokens,
                "kv_cache_memory_bytes": self.runtime.kv_cache_memory_bytes,
                "prefix_caching": self.runtime.enable_prefix_caching,
                "chunked_prefill": self.runtime.enable_chunked_prefill,
                "async_scheduling": self.runtime.async_scheduling,
                "eager_execution": self.runtime.enforce_eager,
                "compilation_requested": not self.runtime.enforce_eager,
                "cuda_graph_capture_requested": not self.runtime.enforce_eager,
                "custom_all_reduce_requested": not self.runtime.disable_custom_all_reduce,
                "marlin_use_atomic_add": self.runtime.marlin_use_atomic_add,
                "language_model_only": self.runtime.language_model_only,
            },
            "validation": {
                "model_runtime": self.validation.model_runtime,
                "responses_api": self.validation.responses_api,
                "long_context": self.validation.long_context,
                "maximum_verified_model_len": self.validation.maximum_verified_model_len,
            },
        }

    def _modality_capabilities(self) -> dict[str, Any]:
        multimodal = self.multimodal

        def validation(value: ModalityValidation) -> dict[str, object]:
            return {
                "status": value.status,
                "native_processor": value.native_processor,
                "native_engine": value.native_engine,
                "responses_api": value.responses_api,
                "context_ladder": value.context_ladder,
                "maximum_verified_context_tokens": value.maximum_verified_context_tokens,
            }

        image = multimodal.image
        audio = multimodal.audio
        mixed = multimodal.mixed_media
        research = multimodal.research_manifest
        promotion = multimodal.promotion_evidence
        return {
            "scope": multimodal.scope,
            "input_modalities": ["image", "audio"] if multimodal.enabled else [],
            "output_modalities": list(multimodal.output_modalities),
            "audio_generation": {
                "supported": False,
                "reason": "Audio generation is a separate project and release contract.",
            },
            "maximum_request_bytes": multimodal.maximum_request_bytes,
            "maximum_context_tokens": multimodal.maximum_context_tokens,
            "minimum_text_context_reserve_tokens": (multimodal.minimum_text_context_reserve_tokens),
            "preprocessing": (
                {
                    "image_patch_size": multimodal.processor.image_patch_size,
                    "image_rescale_factor": multimodal.processor.image_rescale_factor,
                    "image_rescale_max_upscaled_long_edge": (
                        multimodal.processor.image_rescale_max_upscaled_long_edge
                    ),
                    "audio_token_duration_seconds": (
                        multimodal.processor.audio_token_duration_seconds
                    ),
                    "audio_samples_per_token": (multimodal.processor.audio_samples_per_token),
                }
                if multimodal.processor is not None
                else None
            ),
            "research_manifest": (
                {"path": research.path, "sha256": research.sha256} if research is not None else None
            ),
            "release_provenance": (
                {
                    "attestation_sha256": promotion.attestation_sha256,
                    "attestation_payload_sha256": promotion.attestation_payload_sha256,
                    "candidate_profile_sha256": promotion.candidate_profile_sha256,
                    "serving_image_uri": promotion.serving_image_uri,
                    "responses_edge_image_uri": promotion.responses_edge_image_uri,
                    "responses_edge_revision": promotion.responses_edge_revision,
                }
                if promotion is not None
                else None
            ),
            "image": {
                "enabled": image.enabled,
                "formats": list(image.formats),
                "mime_types": list(image.mime_types),
                "transports": list(image.transports),
                "maximum_items": image.maximum_items,
                "maximum_base64_characters": image.maximum_base64_characters,
                "maximum_file_bytes": image.maximum_file_bytes,
                "maximum_decoded_bytes": image.maximum_decoded_bytes,
                "maximum_width": image.maximum_width,
                "maximum_height": image.maximum_height,
                "maximum_pixels": image.maximum_pixels,
                "maximum_decompression_ratio": image.maximum_decompression_ratio,
                "maximum_processor_tokens": image.maximum_processor_tokens,
                "pixel_encoding": "8-bit-rgb-or-rgba-noninterlaced",
                "validation": validation(image.validation),
            },
            "audio": {
                "enabled": audio.enabled,
                "formats": list(audio.formats),
                "mime_types": list(audio.mime_types),
                "transports": list(audio.transports),
                "maximum_items": audio.maximum_items,
                "maximum_base64_characters": audio.maximum_base64_characters,
                "maximum_file_bytes": audio.maximum_file_bytes,
                "maximum_decoded_bytes": audio.maximum_decoded_bytes,
                "maximum_duration_seconds": audio.maximum_duration_seconds,
                "maximum_frames": audio.maximum_frames,
                "sample_rates_hz": list(audio.sample_rates_hz),
                "channels": list(audio.channels),
                "sample_width_bytes": list(audio.sample_width_bytes),
                "encoding": "pcm-signed-little-endian",
                "maximum_processor_tokens": audio.maximum_processor_tokens,
                "validation": validation(audio.validation),
            },
            "mixed_media": {
                "enabled": mixed.enabled,
                "maximum_total_items": mixed.maximum_total_items,
                "maximum_total_decoded_bytes": mixed.maximum_total_decoded_bytes,
                "maximum_processor_tokens": mixed.maximum_processor_tokens,
                "validation": validation(mixed.validation),
            },
        }


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ServingProfileError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServingProfileError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ServingProfileError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ServingProfileError(f"{field} must be a number >= {minimum}")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ServingProfileError(f"{field} must be a boolean")
    return value


def _strings(value: object, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ServingProfileError(f"{field} must be {qualifier} of non-empty strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ServingProfileError(f"{field} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise ServingProfileError(f"{field} must not contain duplicates")
    return tuple(value)


def _integers(value: object, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ServingProfileError(f"{field} must be {qualifier} of positive integers")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        raise ServingProfileError(f"{field} must contain only positive integers")
    if len(value) != len(set(value)):
        raise ServingProfileError(f"{field} must not contain duplicates")
    return tuple(value)


def _sha256(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ServingProfileError(f"{field} must be lowercase SHA-256")
    return digest


def _manifest_pin(value: object, field: str) -> ManifestPin:
    record = _object(value, field)
    return ManifestPin(
        path=_string(record.get("path"), f"{field}.path"),
        sha256=_sha256(record.get("sha256"), f"{field}.sha256"),
    )


def _modality_validation(value: object, field: str) -> ModalityValidation:
    record = _object(value, field)
    return ModalityValidation(
        status=_string(record.get("status"), f"{field}.status"),
        native_processor=_string(record.get("native_processor"), f"{field}.native_processor"),
        native_engine=_string(record.get("native_engine"), f"{field}.native_engine"),
        responses_api=_string(record.get("responses_api"), f"{field}.responses_api"),
        context_ladder=_string(record.get("context_ladder"), f"{field}.context_ladder"),
        maximum_verified_context_tokens=_integer(
            record.get("maximum_verified_context_tokens"),
            f"{field}.maximum_verified_context_tokens",
            minimum=0,
        ),
    )


def _unvalidated_modality() -> ModalityValidation:
    return ModalityValidation(
        status="not-configured",
        native_processor="not-configured",
        native_engine="not-configured",
        responses_api="not-configured",
        context_ladder="not-configured",
        maximum_verified_context_tokens=0,
    )


def _disabled_multimodal(maximum_request_bytes: int = 10_485_760) -> MultimodalSettings:
    validation = _unvalidated_modality()
    return MultimodalSettings(
        enabled=False,
        scope="image-audio-input-to-text-output",
        output_modalities=("text",),
        maximum_request_bytes=maximum_request_bytes,
        maximum_context_tokens=0,
        minimum_text_context_reserve_tokens=0,
        research_manifest=None,
        processor=None,
        image=ImageInputSettings(
            enabled=False,
            formats=(),
            mime_types=(),
            transports=(),
            maximum_items=0,
            maximum_base64_characters=0,
            maximum_file_bytes=0,
            maximum_decoded_bytes=0,
            maximum_width=0,
            maximum_height=0,
            maximum_pixels=0,
            maximum_decompression_ratio=0.0,
            maximum_processor_tokens=0,
            validation=validation,
        ),
        audio=AudioInputSettings(
            enabled=False,
            formats=(),
            mime_types=(),
            transports=(),
            maximum_items=0,
            maximum_base64_characters=0,
            maximum_file_bytes=0,
            maximum_decoded_bytes=0,
            maximum_duration_seconds=0.0,
            maximum_frames=0,
            sample_rates_hz=(),
            channels=(),
            sample_width_bytes=(),
            maximum_processor_tokens=0,
            validation=validation,
        ),
        mixed_media=MixedMediaSettings(
            enabled=False,
            maximum_total_items=0,
            maximum_total_decoded_bytes=0,
            maximum_processor_tokens=0,
            validation=validation,
        ),
        promotion_evidence=None,
    )


def _load_multimodal(value: object) -> MultimodalSettings:
    if value is None:
        return _disabled_multimodal()
    root = _object(value, "multimodal")
    enabled = _boolean(root.get("enabled"), "multimodal.enabled")
    if not enabled:
        return _disabled_multimodal(
            _integer(
                root.get("maximum_request_bytes", 10_485_760),
                "multimodal.maximum_request_bytes",
            )
        )

    processor = _object(root.get("processor"), "multimodal.processor")
    raw_assets = processor.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ServingProfileError("multimodal.processor.assets must be a non-empty array")
    image = _object(root.get("image"), "multimodal.image")
    audio = _object(root.get("audio"), "multimodal.audio")
    mixed = _object(root.get("mixed_media"), "multimodal.mixed_media")
    raw_promotion = root.get("promotion_evidence")
    promotion: PromotionEvidence | None = None
    if raw_promotion is not None:
        evidence = _object(raw_promotion, "multimodal.promotion_evidence")
        promotion = PromotionEvidence(
            attestation_sha256=_sha256(
                evidence.get("attestation_sha256"),
                "multimodal.promotion_evidence.attestation_sha256",
            ),
            attestation_payload_sha256=_sha256(
                evidence.get("attestation_payload_sha256"),
                "multimodal.promotion_evidence.attestation_payload_sha256",
            ),
            candidate_profile_sha256=_sha256(
                evidence.get("candidate_profile_sha256"),
                "multimodal.promotion_evidence.candidate_profile_sha256",
            ),
            serving_image_uri=_string(
                evidence.get("serving_image_uri"),
                "multimodal.promotion_evidence.serving_image_uri",
            ),
            responses_edge_image_uri=_string(
                evidence.get("responses_edge_image_uri"),
                "multimodal.promotion_evidence.responses_edge_image_uri",
            ),
            responses_edge_revision=_string(
                evidence.get("responses_edge_revision"),
                "multimodal.promotion_evidence.responses_edge_revision",
            ),
        )
    return MultimodalSettings(
        enabled=True,
        scope=_string(root.get("scope"), "multimodal.scope"),
        output_modalities=_strings(root.get("output_modalities"), "multimodal.output_modalities"),
        maximum_request_bytes=_integer(
            root.get("maximum_request_bytes"), "multimodal.maximum_request_bytes"
        ),
        maximum_context_tokens=_integer(
            root.get("maximum_context_tokens"), "multimodal.maximum_context_tokens"
        ),
        minimum_text_context_reserve_tokens=_integer(
            root.get("minimum_text_context_reserve_tokens"),
            "multimodal.minimum_text_context_reserve_tokens",
        ),
        research_manifest=_manifest_pin(
            root.get("research_manifest"), "multimodal.research_manifest"
        ),
        processor=ProcessorSettings(
            assets=tuple(
                _manifest_pin(item, f"multimodal.processor.assets[{index}]")
                for index, item in enumerate(raw_assets)
            ),
            image_token_id=_integer(
                processor.get("image_token_id"), "multimodal.processor.image_token_id"
            ),
            audio_token_id=_integer(
                processor.get("audio_token_id"), "multimodal.processor.audio_token_id"
            ),
            image_placeholder=_string(
                processor.get("image_placeholder"), "multimodal.processor.image_placeholder"
            ),
            audio_placeholder=_string(
                processor.get("audio_placeholder"), "multimodal.processor.audio_placeholder"
            ),
            required_weight_prefixes=_strings(
                processor.get("required_weight_prefixes"),
                "multimodal.processor.required_weight_prefixes",
            ),
            image_patch_size=_integer(
                processor.get("image_patch_size"),
                "multimodal.processor.image_patch_size",
            ),
            image_rescale_factor=_number(
                processor.get("image_rescale_factor"),
                "multimodal.processor.image_rescale_factor",
                minimum=1.0,
            ),
            image_rescale_max_upscaled_long_edge=_integer(
                processor.get("image_rescale_max_upscaled_long_edge"),
                "multimodal.processor.image_rescale_max_upscaled_long_edge",
            ),
            audio_token_duration_seconds=_number(
                processor.get("audio_token_duration_seconds"),
                "multimodal.processor.audio_token_duration_seconds",
            ),
            audio_samples_per_token=_integer(
                processor.get("audio_samples_per_token"),
                "multimodal.processor.audio_samples_per_token",
            ),
        ),
        image=ImageInputSettings(
            enabled=_boolean(image.get("enabled"), "multimodal.image.enabled"),
            formats=_strings(image.get("formats"), "multimodal.image.formats"),
            mime_types=_strings(image.get("mime_types"), "multimodal.image.mime_types"),
            transports=_strings(image.get("transports"), "multimodal.image.transports"),
            maximum_items=_integer(image.get("maximum_items"), "multimodal.image.maximum_items"),
            maximum_base64_characters=_integer(
                image.get("maximum_base64_characters"),
                "multimodal.image.maximum_base64_characters",
            ),
            maximum_file_bytes=_integer(
                image.get("maximum_file_bytes"),
                "multimodal.image.maximum_file_bytes",
            ),
            maximum_decoded_bytes=_integer(
                image.get("maximum_decoded_bytes"),
                "multimodal.image.maximum_decoded_bytes",
            ),
            maximum_width=_integer(image.get("maximum_width"), "multimodal.image.maximum_width"),
            maximum_height=_integer(image.get("maximum_height"), "multimodal.image.maximum_height"),
            maximum_pixels=_integer(image.get("maximum_pixels"), "multimodal.image.maximum_pixels"),
            maximum_decompression_ratio=_number(
                image.get("maximum_decompression_ratio"),
                "multimodal.image.maximum_decompression_ratio",
                minimum=1.0,
            ),
            maximum_processor_tokens=_integer(
                image.get("maximum_processor_tokens"),
                "multimodal.image.maximum_processor_tokens",
            ),
            validation=_modality_validation(image.get("validation"), "multimodal.image.validation"),
        ),
        audio=AudioInputSettings(
            enabled=_boolean(audio.get("enabled"), "multimodal.audio.enabled"),
            formats=_strings(audio.get("formats"), "multimodal.audio.formats"),
            mime_types=_strings(audio.get("mime_types"), "multimodal.audio.mime_types"),
            transports=_strings(audio.get("transports"), "multimodal.audio.transports"),
            maximum_items=_integer(audio.get("maximum_items"), "multimodal.audio.maximum_items"),
            maximum_base64_characters=_integer(
                audio.get("maximum_base64_characters"),
                "multimodal.audio.maximum_base64_characters",
            ),
            maximum_file_bytes=_integer(
                audio.get("maximum_file_bytes"),
                "multimodal.audio.maximum_file_bytes",
            ),
            maximum_decoded_bytes=_integer(
                audio.get("maximum_decoded_bytes"),
                "multimodal.audio.maximum_decoded_bytes",
            ),
            maximum_duration_seconds=_number(
                audio.get("maximum_duration_seconds"),
                "multimodal.audio.maximum_duration_seconds",
            ),
            maximum_frames=_integer(audio.get("maximum_frames"), "multimodal.audio.maximum_frames"),
            sample_rates_hz=_integers(
                audio.get("sample_rates_hz"), "multimodal.audio.sample_rates_hz"
            ),
            channels=_integers(audio.get("channels"), "multimodal.audio.channels"),
            sample_width_bytes=_integers(
                audio.get("sample_width_bytes"), "multimodal.audio.sample_width_bytes"
            ),
            maximum_processor_tokens=_integer(
                audio.get("maximum_processor_tokens"),
                "multimodal.audio.maximum_processor_tokens",
            ),
            validation=_modality_validation(audio.get("validation"), "multimodal.audio.validation"),
        ),
        mixed_media=MixedMediaSettings(
            enabled=_boolean(mixed.get("enabled"), "multimodal.mixed_media.enabled"),
            maximum_total_items=_integer(
                mixed.get("maximum_total_items"),
                "multimodal.mixed_media.maximum_total_items",
            ),
            maximum_total_decoded_bytes=_integer(
                mixed.get("maximum_total_decoded_bytes"),
                "multimodal.mixed_media.maximum_total_decoded_bytes",
            ),
            maximum_processor_tokens=_integer(
                mixed.get("maximum_processor_tokens"),
                "multimodal.mixed_media.maximum_processor_tokens",
            ),
            validation=_modality_validation(
                mixed.get("validation"), "multimodal.mixed_media.validation"
            ),
        ),
        promotion_evidence=promotion,
    )


def _load_patches(value: object) -> tuple[RuntimePatch, ...]:
    if not isinstance(value, list) or not value:
        raise ServingProfileError("patches must be a non-empty array")
    patches: list[RuntimePatch] = []
    for index, item in enumerate(value):
        record = _object(item, f"patches[{index}]")
        digest = _string(record.get("sha256"), f"patches[{index}].sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ServingProfileError(f"patches[{index}].sha256 must be lowercase SHA-256")
        patches.append(
            RuntimePatch(
                path=_string(record.get("path"), f"patches[{index}].path"),
                sha256=digest,
            )
        )
    return tuple(patches)


def _validate_invariants(profile: ServingProfile) -> None:
    if profile.schema_version != "1.0.0":
        raise ServingProfileError("schema_version must be 1.0.0")
    if profile.kind != "inkling-responses-serving-profile":
        raise ServingProfileError("kind must be inkling-responses-serving-profile")
    if profile.api.primary_protocol != "responses":
        raise ServingProfileError("only the Responses protocol is supported")
    if profile.api.responses_route != "/v1/responses":
        raise ServingProfileError("api.routes.responses must be /v1/responses")
    if profile.api.streaming_terminal_event != "response.completed":
        raise ServingProfileError("streaming terminal event must be response.completed")
    if profile.api.chat_completions_contract:
        raise ServingProfileError("Chat Completions must not be declared as a service contract")
    if profile.runtime.tensor_parallel_size != 4:
        raise ServingProfileError("the checkpoint contract requires tensor_parallel_size=4")
    if profile.runtime.expert_parallel_size != 1:
        raise ServingProfileError("the validated placement requires expert_parallel_size=1")
    if profile.runtime.dtype != "bfloat16":
        raise ServingProfileError("the W8A16 runtime requires bfloat16 activations")
    if profile.runtime.max_model_len > 1_048_576:
        raise ServingProfileError("max_model_len exceeds Inkling's configured 1M limit")
    if not profile.model.artifact_uri.startswith("gs://"):
        raise ServingProfileError("model.artifact_uri must be a gs:// URI")
    manifest_digest = profile.model.conversion_manifest_sha256
    if len(manifest_digest) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_digest
    ):
        raise ServingProfileError("model.conversion_manifest_sha256 must be lowercase SHA-256")
    if profile.server.port > 65_535:
        raise ServingProfileError("server.port must be <= 65535")
    if profile.runtime.max_num_batched_tokens > profile.runtime.max_model_len:
        raise ServingProfileError("max_num_batched_tokens cannot exceed max_model_len")
    if profile.runtime.max_model_len > 2_048 and not profile.runtime.enable_chunked_prefill:
        raise ServingProfileError("long-context profiles require chunked prefill")
    performance_modes = {
        "conservative",
        "agent-latency-candidate",
        "atlas-stability-baseline",
        "atlas-throughput-candidate",
        "long-context-latency-candidate",
    }
    if profile.runtime.performance_mode not in performance_modes:
        raise ServingProfileError(
            "runtime.performance_mode must be one of " + ", ".join(sorted(performance_modes))
        )
    if (
        profile.runtime.performance_mode == "conservative"
        and profile.runtime.max_model_len > 2_048
        and profile.runtime.max_num_seqs != 1
    ):
        raise ServingProfileError("unvalidated long-context profiles must remain batch-one")
    if profile.runtime.performance_mode != "conservative" and profile.status != (
        "projected-unvalidated"
    ):
        raise ServingProfileError("performance candidates must remain projected-unvalidated")
    if profile.runtime.performance_mode == "agent-latency-candidate" and (
        profile.runtime.max_model_len > 65_536 or profile.runtime.max_num_seqs > 2
    ):
        raise ServingProfileError("agent latency candidates are limited to 64K and two sequences")
    if profile.runtime.performance_mode in {
        "atlas-stability-baseline",
        "atlas-throughput-candidate",
    } and (profile.runtime.max_model_len > 32_768 or profile.runtime.max_num_seqs < 2):
        raise ServingProfileError(
            "Atlas profiles require multiple sequences and at most 32K context"
        )
    if profile.runtime.performance_mode == "long-context-latency-candidate" and (
        profile.runtime.max_model_len <= 65_536 or profile.runtime.max_num_seqs != 1
    ):
        raise ServingProfileError(
            "long-context latency candidates require batch one and more than 64K context"
        )
    if profile.runtime.marlin_use_atomic_add:
        raise ServingProfileError(
            "Marlin atomic-add reduction is ineffective for BF16 on the A100 SM80 contract"
        )
    if profile.memory_projection.allocated_kv_cache_bytes != profile.runtime.kv_cache_memory_bytes:
        raise ServingProfileError("memory projection must match runtime KV allocation")
    if (
        profile.memory_projection.estimated_required_kv_cache_bytes
        > profile.memory_projection.allocated_kv_cache_bytes
    ):
        raise ServingProfileError("KV allocation is below the projected batch-one requirement")
    if profile.validation.maximum_verified_model_len > profile.runtime.max_model_len:
        raise ServingProfileError("verified model length cannot exceed configured model length")
    multimodal = profile.multimodal
    if multimodal.scope != "image-audio-input-to-text-output":
        raise ServingProfileError("multimodal.scope must be image-audio-input-to-text-output")
    if multimodal.output_modalities != ("text",):
        raise ServingProfileError("multimodal output_modalities must contain only text")
    if multimodal.enabled == profile.runtime.language_model_only:
        raise ServingProfileError(
            "multimodal.enabled must be the inverse of runtime.language_model_only"
        )
    if not multimodal.enabled:
        return
    if not multimodal.image.enabled or not multimodal.audio.enabled:
        raise ServingProfileError("multimodal profiles must enable image and audio independently")
    if multimodal.research_manifest is None or multimodal.processor is None:
        raise ServingProfileError(
            "multimodal profiles require research_manifest and processor provenance"
        )
    patch_paths = {patch.path for patch in profile.patches}
    required_multimodal_patches = {
        "patches/vllm/0005-responses-input-audio-content.patch",
        "patches/vllm/0006-inkling-multimodal-profile-bounds.patch",
    }
    if not required_multimodal_patches.issubset(patch_paths):
        raise ServingProfileError(
            "multimodal profiles require the Responses-audio and profiling-bound patches"
        )
    if multimodal.maximum_context_tokens != profile.runtime.max_model_len:
        raise ServingProfileError("multimodal context limit must match runtime max_model_len")
    if multimodal.minimum_text_context_reserve_tokens >= multimodal.maximum_context_tokens:
        raise ServingProfileError("multimodal text context reserve exhausts the context window")
    if multimodal.image.transports != ("data-url",):
        raise ServingProfileError("initial image transport must be data-url only")
    if multimodal.audio.transports != ("base64",):
        raise ServingProfileError("initial audio transport must be base64 only")
    if multimodal.image.maximum_base64_characters < (
        4 * ((multimodal.image.maximum_file_bytes + 2) // 3)
    ):
        raise ServingProfileError("image base64 limit cannot encode the maximum file size")
    if multimodal.audio.maximum_base64_characters < (
        4 * ((multimodal.audio.maximum_file_bytes + 2) // 3)
    ):
        raise ServingProfileError("audio base64 limit cannot encode the maximum file size")
    if multimodal.mixed_media.maximum_total_items < (
        multimodal.image.maximum_items + multimodal.audio.maximum_items
    ):
        raise ServingProfileError("mixed-media item limit is below the per-modality limits")
    processor_paths = [pin.path for pin in multimodal.processor.assets]
    if len(processor_paths) != len(set(processor_paths)):
        raise ServingProfileError("multimodal processor assets contain duplicate paths")
    for pin in (*multimodal.processor.assets, multimodal.research_manifest):
        relative = Path(pin.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ServingProfileError(f"unsafe multimodal provenance path: {pin.path!r}")
    processor = multimodal.processor
    if len(multimodal.audio.sample_rates_hz) != 1:
        raise ServingProfileError("initial audio profile must pin exactly one sample rate")
    sample_rate = multimodal.audio.sample_rates_hz[0]
    expected_samples_per_token = processor.audio_token_duration_seconds * sample_rate
    if (
        not expected_samples_per_token.is_integer()
        or int(expected_samples_per_token) != processor.audio_samples_per_token
    ):
        raise ServingProfileError("audio samples-per-token differs from preprocessing")
    long_edge = max(multimodal.image.maximum_width, multimodal.image.maximum_height)
    target_long_edge = min(
        float(long_edge) * processor.image_rescale_factor,
        float(max(processor.image_rescale_max_upscaled_long_edge, long_edge)),
    )
    ratio = target_long_edge / long_edge
    processed_width = max(1, math.floor(multimodal.image.maximum_width * ratio + 0.5))
    processed_height = max(1, math.floor(multimodal.image.maximum_height * ratio + 0.5))
    image_tokens = (
        (processed_height + processor.image_patch_size - 1) // processor.image_patch_size
    ) * (processed_width // processor.image_patch_size + 1)
    if image_tokens != multimodal.image.maximum_processor_tokens:
        raise ServingProfileError("image processor-token limit differs from preprocessing")
    audio_tokens = (
        multimodal.audio.maximum_frames + processor.audio_samples_per_token - 1
    ) // processor.audio_samples_per_token
    if audio_tokens != multimodal.audio.maximum_processor_tokens:
        raise ServingProfileError("audio processor-token limit differs from preprocessing")
    for name, maximum in (
        ("image", multimodal.image.maximum_processor_tokens),
        ("audio", multimodal.audio.maximum_processor_tokens),
        ("mixed_media", multimodal.mixed_media.maximum_processor_tokens),
    ):
        if maximum + multimodal.minimum_text_context_reserve_tokens >= (
            multimodal.maximum_context_tokens
        ):
            raise ServingProfileError(
                f"multimodal.{name} processor budget leaves no output-token capacity"
            )
    allowed_states = {"not-configured", "unvalidated", "validated", "failed"}
    has_validated_modality = False
    for name, evidence in (
        ("image", multimodal.image.validation),
        ("audio", multimodal.audio.validation),
        ("mixed_media", multimodal.mixed_media.validation),
    ):
        states = (
            evidence.status,
            evidence.native_processor,
            evidence.native_engine,
            evidence.responses_api,
            evidence.context_ladder,
        )
        if any(state not in allowed_states for state in states):
            raise ServingProfileError(f"multimodal.{name}.validation contains an invalid state")
        if evidence.maximum_verified_context_tokens > profile.runtime.max_model_len:
            raise ServingProfileError(
                f"multimodal.{name} verified context exceeds runtime max_model_len"
            )
        component_states = states[1:]
        if evidence.status == "validated" and any(
            state != "validated" for state in component_states
        ):
            raise ServingProfileError(
                f"multimodal.{name} cannot be validated before every component gate"
            )
        if evidence.status == "validated" and evidence.maximum_verified_context_tokens <= 0:
            raise ServingProfileError(
                f"multimodal.{name} validated context maximum must be positive"
            )
        has_validated_modality = has_validated_modality or evidence.status == "validated"
    if has_validated_modality and multimodal.promotion_evidence is None:
        raise ServingProfileError(
            "validated multimodal capability states require detached promotion evidence"
        )
    promotion = multimodal.promotion_evidence
    if promotion is not None:
        for field, uri in (
            ("serving_image_uri", promotion.serving_image_uri),
            ("responses_edge_image_uri", promotion.responses_edge_image_uri),
        ):
            prefix, separator, digest = uri.rpartition("@sha256:")
            if (
                not prefix
                or not separator
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ServingProfileError(
                    f"multimodal.promotion_evidence.{field} must be an immutable image URI"
                )


def load_serving_profile(path: Path) -> ServingProfile:
    """Load one profile and reject missing, mistyped, or unsafe combinations."""

    resolved = path.expanduser().resolve()
    try:
        profile_bytes = resolved.read_bytes()
        root = _object(json.loads(profile_bytes), "profile")
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingProfileError(f"cannot load serving profile {resolved}: {exc}") from exc

    model = _object(root.get("model"), "model")
    api = _object(root.get("api"), "api")
    routes = _object(api.get("routes"), "api.routes")
    streaming = _object(api.get("streaming"), "api.streaming")
    structured = _object(api.get("structured_outputs"), "api.structured_outputs")
    response_store = _object(api.get("response_store"), "api.response_store")
    server = _object(root.get("server"), "server")
    runtime = _object(root.get("runtime"), "runtime")
    validation = _object(root.get("validation"), "validation")
    projection = _object(root.get("memory_projection"), "memory_projection")

    profile = ServingProfile(
        path=resolved,
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        schema_version=_string(root.get("schema_version"), "schema_version"),
        kind=_string(root.get("kind"), "kind"),
        profile_id=_string(root.get("profile_id"), "profile_id"),
        status=_string(root.get("status"), "status"),
        description=_string(root.get("description"), "description"),
        model=ModelSettings(
            served_model_name=_string(model.get("served_model_name"), "model.served_model_name"),
            checkpoint_id=_string(model.get("checkpoint_id"), "model.checkpoint_id"),
            quantization=_string(model.get("quantization"), "model.quantization"),
            artifact_uri=_string(model.get("artifact_uri"), "model.artifact_uri"),
            conversion_manifest_sha256=_string(
                model.get("conversion_manifest_sha256"),
                "model.conversion_manifest_sha256",
            ),
        ),
        api=ApiSettings(
            primary_protocol=_string(api.get("primary_protocol"), "api.primary_protocol"),
            models_route=_string(routes.get("models"), "api.routes.models"),
            responses_route=_string(routes.get("responses"), "api.routes.responses"),
            capabilities_route=_string(routes.get("capabilities"), "api.routes.capabilities"),
            streaming_terminal_event=_string(
                streaming.get("terminal_event"), "api.streaming.terminal_event"
            ),
            structured_json_schema=_boolean(
                structured.get("json_schema"), "api.structured_outputs.json_schema"
            ),
            response_store_enabled=_boolean(
                response_store.get("enabled"), "api.response_store.enabled"
            ),
            chat_completions_contract=_boolean(
                api.get("chat_completions_contract"), "api.chat_completions_contract"
            ),
        ),
        server=ServerSettings(
            host=_string(server.get("host"), "server.host"),
            port=_integer(server.get("port"), "server.port"),
        ),
        runtime=RuntimeSettings(
            vllm_version=_string(runtime.get("vllm_version"), "runtime.vllm_version"),
            vllm_commit=_string(runtime.get("vllm_commit"), "runtime.vllm_commit"),
            tensor_parallel_size=_integer(
                runtime.get("tensor_parallel_size"), "runtime.tensor_parallel_size"
            ),
            expert_parallel_size=_integer(
                runtime.get("expert_parallel_size"), "runtime.expert_parallel_size"
            ),
            dtype=_string(runtime.get("dtype"), "runtime.dtype"),
            tokenizer_mode=_string(runtime.get("tokenizer_mode"), "runtime.tokenizer_mode"),
            max_model_len=_integer(runtime.get("max_model_len"), "runtime.max_model_len"),
            max_num_seqs=_integer(runtime.get("max_num_seqs"), "runtime.max_num_seqs"),
            max_num_batched_tokens=_integer(
                runtime.get("max_num_batched_tokens"), "runtime.max_num_batched_tokens"
            ),
            block_size=_integer(runtime.get("block_size"), "runtime.block_size"),
            kv_cache_memory_bytes=_integer(
                runtime.get("kv_cache_memory_bytes"), "runtime.kv_cache_memory_bytes"
            ),
            cpu_offload_gib=_number(runtime.get("cpu_offload_gib"), "runtime.cpu_offload_gib"),
            enforce_eager=_boolean(runtime.get("enforce_eager"), "runtime.enforce_eager"),
            enable_prefix_caching=_boolean(
                runtime.get("enable_prefix_caching"), "runtime.enable_prefix_caching"
            ),
            enable_chunked_prefill=_boolean(
                runtime.get("enable_chunked_prefill"), "runtime.enable_chunked_prefill"
            ),
            async_scheduling=_boolean(runtime.get("async_scheduling"), "runtime.async_scheduling"),
            language_model_only=_boolean(
                runtime.get("language_model_only"), "runtime.language_model_only"
            ),
            disable_custom_all_reduce=_boolean(
                runtime.get("disable_custom_all_reduce"), "runtime.disable_custom_all_reduce"
            ),
            distributed_executor_backend=_string(
                runtime.get("distributed_executor_backend"),
                "runtime.distributed_executor_backend",
            ),
            reasoning_parser=_string(runtime.get("reasoning_parser"), "runtime.reasoning_parser"),
            tool_call_parser=_string(runtime.get("tool_call_parser"), "runtime.tool_call_parser"),
            enable_auto_tool_choice=_boolean(
                runtime.get("enable_auto_tool_choice"), "runtime.enable_auto_tool_choice"
            ),
            performance_mode=_string(
                runtime.get("performance_mode", "conservative"),
                "runtime.performance_mode",
            ),
            marlin_use_atomic_add=_boolean(
                runtime.get("marlin_use_atomic_add", False),
                "runtime.marlin_use_atomic_add",
            ),
            seed=_integer(runtime.get("seed"), "runtime.seed", minimum=0),
        ),
        validation=ValidationSettings(
            model_runtime=_string(validation.get("model_runtime"), "validation.model_runtime"),
            responses_api=_string(validation.get("responses_api"), "validation.responses_api"),
            long_context=_string(validation.get("long_context"), "validation.long_context"),
            maximum_verified_model_len=_integer(
                validation.get("maximum_verified_model_len"),
                "validation.maximum_verified_model_len",
            ),
        ),
        memory_projection=MemoryProjection(
            estimated_required_kv_cache_bytes=_integer(
                projection.get("estimated_required_kv_cache_bytes"),
                "memory_projection.estimated_required_kv_cache_bytes",
            ),
            allocated_kv_cache_bytes=_integer(
                projection.get("allocated_kv_cache_bytes"),
                "memory_projection.allocated_kv_cache_bytes",
            ),
            evidence=_string(projection.get("evidence"), "memory_projection.evidence"),
        ),
        multimodal=_load_multimodal(root.get("multimodal")),
        patches=_load_patches(root.get("patches")),
    )
    _validate_invariants(profile)
    return profile
