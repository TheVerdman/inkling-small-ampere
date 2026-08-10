"""Strict, dependency-free loading for Inkling serving profiles."""

from __future__ import annotations

import json
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
class ServingProfile:
    """Validated launch and capability configuration for one service shape."""

    path: Path
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
    patches: tuple[RuntimePatch, ...]

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
            "schema_version": "1.0.0",
            "service": "inkling-small-ampere",
            "profile_id": self.profile_id,
            "profile_status": self.status,
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
            "runtime": {
                "vllm_version": self.runtime.vllm_version,
                "vllm_commit": self.runtime.vllm_commit,
                "tensor_parallel_size": self.runtime.tensor_parallel_size,
                "dtype": self.runtime.dtype,
                "max_model_len": self.runtime.max_model_len,
                "max_num_seqs": self.runtime.max_num_seqs,
                "max_num_batched_tokens": self.runtime.max_num_batched_tokens,
                "kv_cache_memory_bytes": self.runtime.kv_cache_memory_bytes,
                "prefix_caching": self.runtime.enable_prefix_caching,
                "chunked_prefill": self.runtime.enable_chunked_prefill,
                "async_scheduling": self.runtime.async_scheduling,
                "language_model_only": self.runtime.language_model_only,
            },
            "validation": {
                "model_runtime": self.validation.model_runtime,
                "responses_api": self.validation.responses_api,
                "long_context": self.validation.long_context,
                "maximum_verified_model_len": self.validation.maximum_verified_model_len,
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
    if profile.runtime.max_model_len > 2_048 and profile.runtime.max_num_seqs != 1:
        raise ServingProfileError("unvalidated long-context profiles must remain batch-one")
    if profile.memory_projection.allocated_kv_cache_bytes != profile.runtime.kv_cache_memory_bytes:
        raise ServingProfileError("memory projection must match runtime KV allocation")
    if (
        profile.memory_projection.estimated_required_kv_cache_bytes
        > profile.memory_projection.allocated_kv_cache_bytes
    ):
        raise ServingProfileError("KV allocation is below the projected batch-one requirement")
    if profile.validation.maximum_verified_model_len > profile.runtime.max_model_len:
        raise ServingProfileError("verified model length cannot exceed configured model length")


def load_serving_profile(path: Path) -> ServingProfile:
    """Load one profile and reject missing, mistyped, or unsafe combinations."""

    resolved = path.expanduser().resolve()
    try:
        root = _object(json.loads(resolved.read_text()), "profile")
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
        patches=_load_patches(root.get("patches")),
    )
    _validate_invariants(profile)
    return profile
