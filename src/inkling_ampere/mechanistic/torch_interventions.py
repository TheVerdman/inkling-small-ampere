"""Torch/vLLM treatment hooks for manifest-bound causal experiments."""

from __future__ import annotations

import contextlib
import importlib
import types
from collections.abc import Callable, Mapping
from typing import Any, cast

from inkling_ampere.mechanistic.contracts import (
    InterventionKind,
    MechanisticContractError,
    ModuleKind,
)
from inkling_ampere.mechanistic.interventions import (
    InterventionManifest,
    InterventionRuntime,
    InterventionSpec,
    TorchParameterCounterfactual,
)

TensorSourceLoader = Callable[[InterventionSpec], Any]
KVAdapter = Callable[[Any, InterventionSpec, TensorSourceLoader], Any]


class TorchInterventionController:
    """Install actual treatment hooks; observation-only code never imports this module."""

    def __init__(
        self,
        *,
        model: Any,
        manifest: InterventionManifest,
        runtime: InterventionRuntime,
        treatment_run_id: str,
        tensor_source_loader: TensorSourceLoader,
        kv_adapter: KVAdapter | None = None,
    ) -> None:
        self.model = model
        self.manifest = manifest
        self.runtime = runtime
        self.treatment_run_id = treatment_run_id
        self.tensor_source_loader = tensor_source_loader
        self.kv_adapter = kv_adapter
        self.torch: Any = importlib.import_module("torch")
        self.handles: list[Any] = []
        self.originals: list[tuple[Any, str, bool, Any]] = []
        self.transactions = contextlib.ExitStack()
        self.positions: list[int] = []
        self.installed = False

    def install(self) -> None:
        self.runtime.require_treatment(self.treatment_run_id)
        if self.installed:
            raise MechanisticContractError("intervention controller is already installed")

        def position_hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            if positions is None or not hasattr(positions, "detach"):
                raise MechanisticContractError("treatment hook cannot resolve token positions")
            self.positions = [
                int(position) for position in positions.detach().reshape(-1).cpu().tolist()
            ]

        try:
            self.handles.append(
                self.model.register_forward_pre_hook(position_hook, with_kwargs=True)
            )
            for intervention in self.manifest.interventions:
                self._install_one(intervention)
        except BaseException:
            self.remove()
            raise
        self.installed = True
        self.runtime.register_cleanup(
            f"torch intervention controller {self.manifest.manifest_id}", self.remove
        )

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        for owner, attribute, had_instance_value, instance_value in reversed(self.originals):
            if had_instance_value:
                setattr(owner, attribute, instance_value)
            else:
                delattr(owner, attribute)
        self.originals.clear()
        self.transactions.close()
        self.installed = False

    def _install_one(self, intervention: InterventionSpec) -> None:
        if intervention.kind in {
            InterventionKind.EXPERT_KNOCKOUT,
            InterventionKind.EXPERT_ATTENUATION,
            InterventionKind.EXPERT_AMPLIFICATION,
            InterventionKind.EXPERT_REROUTE,
            InterventionKind.ROUTE_FREEZE,
        }:
            self._install_route(intervention)
            return
        if intervention.kind in {
            InterventionKind.ACTIVATION_PATCH,
            InterventionKind.RESIDUAL_STEERING,
            InterventionKind.VECTOR_INJECTION,
            InterventionKind.OUTPUT_ABLATION,
            InterventionKind.MODALITY_EMBEDDING_SWAP,
        }:
            self._install_activation(intervention)
            return
        if intervention.kind is InterventionKind.LOGIT_BIAS:
            self._install_logit_bias(intervention)
            return
        if intervention.kind in {
            InterventionKind.QUANTIZATION_SCALE,
            InterventionKind.PRECISION_RESTORATION,
        }:
            self._install_parameter_counterfactual(intervention)
            return
        if intervention.kind in {InterventionKind.KV_PATCH, InterventionKind.CONTEXT_MASK}:
            if self.kv_adapter is None:
                raise MechanisticContractError(
                    f"{intervention.kind.value} requires an explicit pinned-runtime KV adapter"
                )
            handle = self.kv_adapter(self.model, intervention, self.tensor_source_loader)
            self.handles.append(handle)
            return
        raise MechanisticContractError(f"unsupported intervention {intervention.kind.value}")

    def _install_route(self, intervention: InterventionSpec) -> None:
        for layer_index in intervention.layers:
            layer = self._layer(layer_index)
            gate = getattr(layer.mlp, "gate", None)
            if gate is None:
                raise MechanisticContractError(
                    f"layer {layer_index} has no MoE gate for route intervention"
                )
            total_experts = int(gate.n_total_experts)
            if any(expert >= total_experts for expert in intervention.expert_ids):
                raise MechanisticContractError(
                    f"route intervention expert id exceeds layer {layer_index} expert count"
                )
            mapping = intervention.parameters.get("mapping", {})
            if isinstance(mapping, Mapping) and any(
                int(source) >= total_experts or int(cast(int, target)) >= total_experts
                for source, target in mapping.items()
            ):
                raise MechanisticContractError(
                    f"route intervention mapping exceeds layer {layer_index} expert count"
                )
            original = gate.select_experts

            def wrapped(
                _gate: Any,
                gating_output: Any,
                *,
                original: Any = original,
                intervention: InterventionSpec = intervention,
                total_experts: int = total_experts,
            ) -> Any:
                weights, ids = original(gating_output)
                treated_weights, treated_ids = self._treat_routes(
                    weights,
                    ids,
                    intervention,
                    total_experts=total_experts,
                )
                return treated_weights, treated_ids

            instance_values = getattr(gate, "__dict__", {})
            self.originals.append(
                (
                    gate,
                    "select_experts",
                    "select_experts" in instance_values,
                    instance_values.get("select_experts"),
                )
            )
            gate.select_experts = types.MethodType(wrapped, gate)

    def _treat_routes(
        self,
        weights: Any,
        ids: Any,
        intervention: InterventionSpec,
        *,
        total_experts: int,
    ) -> tuple[Any, Any]:
        treated_weights = weights.clone()
        treated_ids = ids.clone()
        rows = treated_weights.reshape(-1, treated_weights.shape[-1])
        id_rows = treated_ids.reshape(-1, treated_ids.shape[-1])
        selected_rows = self._selected_rows(rows.shape[0], intervention, rows.device)
        expert_tensor = self.torch.tensor(
            list(intervention.expert_ids), dtype=id_rows.dtype, device=id_rows.device
        )
        if intervention.kind is InterventionKind.ROUTE_FREEZE:
            source = self.tensor_source_loader(intervention)
            if (
                not isinstance(source, Mapping)
                or "weights" not in source
                or "expert_ids" not in source
            ):
                raise MechanisticContractError(
                    "route-freeze source loader must return weights and expert_ids"
                )
            frozen_weights = source["weights"].to(rows)
            frozen_ids = source["expert_ids"].to(id_rows)
            if frozen_weights.shape != rows.shape or frozen_ids.shape != id_rows.shape:
                raise MechanisticContractError("frozen route tensors do not match target shape")
            if bool(((frozen_ids < 0) | (frozen_ids >= total_experts)).any().item()):
                raise MechanisticContractError("frozen route contains an out-of-range expert")
            rows[selected_rows] = frozen_weights[selected_rows]
            id_rows[selected_rows] = frozen_ids[selected_rows]
            return treated_weights, treated_ids
        if intervention.kind is InterventionKind.EXPERT_REROUTE:
            mapping = intervention.parameters.get("mapping")
            if not isinstance(mapping, Mapping):
                raise MechanisticContractError("expert reroute requires mapping")
            for source, target in mapping.items():
                source_id = int(source)
                if not isinstance(target, int) or isinstance(target, bool):
                    raise MechanisticContractError("reroute targets must be integers")
                mask = id_rows[selected_rows] == source_id
                id_rows[selected_rows] = self.torch.where(
                    mask,
                    self.torch.full_like(id_rows[selected_rows], target),
                    id_rows[selected_rows],
                )
            return treated_weights, treated_ids
        if intervention.kind is InterventionKind.EXPERT_KNOCKOUT:
            factor = 0.0
        else:
            factor = float(cast(int | float, intervention.parameters["factor"]))
        row_ids = id_rows[selected_rows]
        mask = (row_ids.unsqueeze(-1) == expert_tensor).any(dim=-1)
        original_mass = rows[selected_rows].sum(dim=-1, keepdim=True)
        rows[selected_rows] = self.torch.where(
            mask, rows[selected_rows] * factor, rows[selected_rows]
        )
        if bool(intervention.parameters.get("renormalize", False)):
            new_mass = rows[selected_rows].sum(dim=-1, keepdim=True)
            if bool((new_mass <= 0).any().item()):
                raise MechanisticContractError("route treatment produced zero mass")
            rows[selected_rows] *= original_mass / new_mass
        return treated_weights, treated_ids

    def _install_activation(self, intervention: InterventionSpec) -> None:
        for layer_index in intervention.layers:
            module = self._module_for(intervention.module_kind, layer_index, intervention)

            def hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                intervention: InterventionSpec = intervention,
            ) -> Any:
                return self._replace_activation_output(output, intervention)

            self.handles.append(module.register_forward_hook(hook))

    def _replace_activation_output(self, output: Any, intervention: InterventionSpec) -> Any:
        if isinstance(output, self.torch.Tensor) and output.is_floating_point():
            return self._treat_activation(output, intervention)
        if isinstance(output, tuple):
            values = list(output)
            for index, value in enumerate(values):
                if isinstance(value, self.torch.Tensor) and value.is_floating_point():
                    values[index] = self._treat_activation(value, intervention)
                    return tuple(values)
        raise MechanisticContractError(
            f"{intervention.intervention_id} hook found no floating activation output"
        )

    def _treat_activation(self, activation: Any, intervention: InterventionSpec) -> Any:
        treated = activation.clone()
        rows = treated.reshape(-1, treated.shape[-1])
        selected = self._selected_rows(rows.shape[0], intervention, rows.device)
        if intervention.kind is InterventionKind.OUTPUT_ABLATION:
            rows[selected] = 0
            return treated
        source = self.tensor_source_loader(intervention)
        if intervention.kind in {
            InterventionKind.ACTIVATION_PATCH,
            InterventionKind.MODALITY_EMBEDDING_SWAP,
        }:
            replacement = source.to(rows)
            if replacement.shape == rows.shape:
                rows[selected] = replacement[selected]
            elif replacement.shape == rows[selected].shape:
                rows[selected] = replacement
            else:
                raise MechanisticContractError("activation patch source shape mismatch")
            return treated
        vector = source.to(rows).reshape(-1)
        if vector.shape[0] != rows.shape[-1]:
            raise MechanisticContractError("steering vector width mismatch")
        factor = float(cast(int | float, intervention.parameters["factor"]))
        rows[selected] += factor * vector
        return treated

    def _install_logit_bias(self, intervention: InterventionSpec) -> None:
        raw_biases = intervention.parameters.get("token_biases")
        if not isinstance(raw_biases, Mapping):
            raise MechanisticContractError("logit-bias requires token_biases mapping")
        biases: dict[int, float] = {}
        for token_id, bias in raw_biases.items():
            if not isinstance(bias, (int, float)) or isinstance(bias, bool):
                raise MechanisticContractError("logit biases must be numeric")
            biases[int(token_id)] = float(bias)
        original = self.model.compute_logits

        def wrapped(_model: Any, hidden_states: Any) -> Any:
            logits = original(hidden_states)
            if logits is None:
                return None
            treated = logits.clone()
            rows = treated.reshape(-1, treated.shape[-1])
            selected = self._selected_rows(rows.shape[0], intervention, rows.device)
            for token_id, bias in biases.items():
                if not 0 <= token_id < rows.shape[-1]:
                    raise MechanisticContractError("logit-bias token id is out of range")
                rows[selected, token_id] += bias
            return treated

        instance_values = getattr(self.model, "__dict__", {})
        self.originals.append(
            (
                self.model,
                "compute_logits",
                "compute_logits" in instance_values,
                instance_values.get("compute_logits"),
            )
        )
        self.model.compute_logits = types.MethodType(wrapped, self.model)

    def _install_parameter_counterfactual(self, intervention: InterventionSpec) -> None:
        parameter_path = intervention.parameters.get("parameter_path")
        if not isinstance(parameter_path, str) or not parameter_path:
            raise MechanisticContractError("parameter counterfactual requires parameter_path")
        parameter = _resolve_attribute(self.model, parameter_path)
        if intervention.kind is InterventionKind.QUANTIZATION_SCALE:
            factor = float(cast(int | float, intervention.parameters["factor"]))
            replacement = parameter.detach() * factor
        else:
            replacement = self.tensor_source_loader(intervention)
        self.transactions.enter_context(TorchParameterCounterfactual(parameter, replacement))

    def _selected_rows(self, row_count: int, intervention: InterventionSpec, device: Any) -> Any:
        if len(self.positions) != row_count:
            observed_positions = len(self.positions)
            raise MechanisticContractError(
                f"treatment row count {row_count} does not align with "
                f"{observed_positions} positions"
            )
        if intervention.token_positions:
            selected_positions = set(intervention.token_positions)
            indices = [
                index
                for index, position in enumerate(self.positions)
                if position in selected_positions
            ]
        else:
            indices = list(range(row_count))
        if not indices:
            return self.torch.empty(0, dtype=self.torch.long, device=device)
        return self.torch.tensor(indices, dtype=self.torch.long, device=device)

    def _layer(self, layer_index: int) -> Any:
        layers = self.model.model.layers
        if not 0 <= layer_index < len(layers):
            raise MechanisticContractError(f"intervention layer {layer_index} is out of range")
        return layers[layer_index]

    def _module_for(
        self, kind: ModuleKind, layer_index: int, intervention: InterventionSpec
    ) -> Any:
        layer = self._layer(layer_index)
        modules = {
            ModuleKind.RESIDUAL_STREAM: layer,
            ModuleKind.ATTENTION_OUTPUT: layer.attn,
            ModuleKind.MLP_OUTPUT: layer.mlp,
            ModuleKind.EXPERT_OUTPUT: getattr(layer.mlp, "experts", None),
            ModuleKind.SHARED_EXPERT_OUTPUT: getattr(layer.mlp, "sink_experts", None),
        }
        if kind in {ModuleKind.MODALITY_ENCODER, ModuleKind.MODALITY_PROJECTION}:
            module_path = intervention.parameters.get("module_path")
            if not isinstance(module_path, str):
                raise MechanisticContractError("modality intervention requires module_path")
            return _resolve_attribute(self.model, module_path)
        module = modules.get(kind)
        if module is None:
            raise MechanisticContractError(
                f"no exact treatment hook for {kind.value} at layer {layer_index}"
            )
        return module


def _resolve_attribute(root: Any, dotted_path: str) -> Any:
    current = root
    for component in dotted_path.split("."):
        if not component or not hasattr(current, component):
            raise MechanisticContractError(f"cannot resolve parameter/module {dotted_path!r}")
        current = getattr(current, component)
    return current
