"""Skill packs: low-rank adapters that attach to a frozen PseudoBrain and detach exactly.

A pack is a set of LoRA factors (Hu et al., "LoRA: Low-Rank Adaptation of Large Language
Models", ICLR 2022) for chosen ``nn.Linear`` weights of a base model, plus a manifest that
names the base it was trained on (its state-dict fingerprint and recipe). Attached, a
weight reads ``W + sum_packs (alpha / r) B A``; the base tensor ``W`` itself is never
written.

Mechanism: ``torch.nn.utils.parametrize``. The adapted weight is a parametrization of the
module's ``weight``, so every code path that reads ``module.weight`` sees it: the folded
output head, the byte-table fold, boundary key folding, banked per-boundary linears, the
fused Triton kernels and ``torch.compile``'d pieces, not only ``module.forward``. The base
``Parameter`` object moves into ``parametrizations.weight.original`` untouched and comes
back as ``module.weight`` when the last pack detaches, so detaching restores the base
bitwise (same tensor, same fingerprint). While packs are attached ``model.state_dict()``
carries the parametrization keys; ``base_state_dict`` gives the plain base.

Serving: ``set_merged(model, True)`` caches ``W + delta`` per weight (one extra copy of the
adapted weights) so a decode step does not recompute ``B A`` at every weight read.

Composition: several packs can be attached at once; their deltas add.

Growth: ``grow_pack`` maps a pack onto a model grown with ``efficiency.growth`` exactly
(B's output rows are replicated like the weight's, A's input columns split like the
weight's, inserted core layers get no adapter), so a pack survives the base's growth.

Pack format (a directory): ``manifest.json`` (schema ``pb_skill_pack_v1``) and
``adapter.pt`` (``{"<module>.A": [r, in], "<module>.B": [out, r]}``, loadable with
``torch.load(weights_only=True)``).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

SKILL_SCHEMA = "pb_skill_pack_v1"
# Default targets: the heavy core's recurrent input and output projections, the local byte
# attention (q/k/v, output, MLP), the local byte decoder MLPs, and the converted TokenCore /
# GraftedCore attention, GDN linear projections, and MoE shared expert projections.
DEFAULT_TARGETS = (
    r"fast\.blocks\.\d+\.(gates|out)",
    r"local_attention\.blocks\.\d+\.(qkv|out|mlp\.gate_up|mlp\.down)",
    r"local_decoder\.blocks\.\d+\.(gate_up|down)",
    r"(?:core\.)?blocks\.\d+\.(q|k|v|out|in_proj|out_proj|shared_gate_up|shared_down)",
    r"(?:core\.)?blocks\.\d+\.mlp\.(gate_proj|up_proj|down_proj|gate_up|down)",
)
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _key(name: str) -> str:
    """ModuleDict key of a pack: its name (no dots; 1-64 of [A-Za-z0-9_-])."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"pack name {name!r}: 1-64 of [A-Za-z0-9_-], starting alphanumeric")
    return name


class LowRankFactors(nn.Module):
    """One pack's factors for one weight: delta = (alpha / rank) * B @ A (B starts at zero)."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float, *, device=None,
                 dtype=None):
        super().__init__()
        self.rank, self.alpha = int(rank), float(alpha)
        self.A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale


class LowRankDelta(nn.Module):
    """The parametrization ``W -> W + sum of the attached packs' deltas`` of one weight."""

    def __init__(self):
        super().__init__()
        self.packs = nn.ModuleDict()
        self.merged: torch.Tensor | None = None
        self.bypass = False          # packs_bypassed(): read the base weight (e.g. a KL reference pass)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.bypass:
            return weight
        if self.merged is not None and not torch.is_grad_enabled():
            return self.merged
        out = weight
        for factors in self.packs.values():
            out = out + factors.delta().to(weight.dtype)
        return out


# ----------------------------------------------------------------------------- base identity

def _delta_of(module: nn.Module) -> LowRankDelta | None:
    if not parametrize.is_parametrized(module, "weight"):
        return None
    for p in module.parametrizations.weight:
        if isinstance(p, LowRankDelta):
            return p
    return None


def base_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The base model's state dict with plain names, whatever packs are attached."""
    out = {}
    for name, value in model.state_dict().items():
        if ".parametrizations." in name:
            if name.endswith(".parametrizations.weight.original"):
                out[name.replace(".parametrizations.weight.original", ".weight")] = value
            continue
        out[name] = value
    return out


def base_fingerprint(model: nn.Module) -> str:
    """``pb7.runtime.fingerprint`` of the base (sorted state-dict bytes), packs excluded."""
    digest = hashlib.sha256()
    for name, value in sorted(base_state_dict(model).items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_recipe(model) -> dict | None:
    from dataclasses import asdict

    if not hasattr(model, "config") or not hasattr(model, "efficiency_options"):
        return None
    return json.loads(json.dumps({"config": asdict(model.config), "options": asdict(model.efficiency_options)}))


# ----------------------------------------------------------------------------- targets

def target_modules(model: nn.Module, patterns=DEFAULT_TARGETS) -> list[str]:
    """Names of the ``nn.Linear`` modules matching ``patterns`` (skipping structurally unused ones)."""
    compiled = [re.compile(p + r"$") for p in patterns]
    inactive = set(getattr(model, "inactive_parameter_names", lambda: set())())
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not any(c.match(name) for c in compiled):
            continue
        weight_name = f"{name}.weight"
        if weight_name in inactive:               # e.g. the core's top-layer out projection
            continue
        names.append(name)
    if not names:
        raise ValueError(f"no linear module matches {list(patterns)}")
    return names


def _module(model: nn.Module, name: str) -> nn.Module:
    return model.get_submodule(name)


# ----------------------------------------------------------------------------- attach / detach

def attached_packs(model: nn.Module) -> list[str]:
    names = []
    for module in model.modules():
        delta = _delta_of(module)
        if delta is not None:
            names.extend(k for k in delta.packs if k not in names)
    return names


def _ensure_delta(module: nn.Module) -> LowRankDelta:
    delta = _delta_of(module)
    if delta is None:
        parametrize.register_parametrization(module, "weight", LowRankDelta(), unsafe=True)
        delta = _delta_of(module)
    return delta


def add_pack(model: nn.Module, name: str, *, rank: int = 8, alpha: float = 16.0, targets=DEFAULT_TARGETS,
             tensors: dict[str, torch.Tensor] | None = None, trainable: bool = True) -> list[str]:
    """Attach pack ``name``: fresh factors (B = 0: exact at attach) or ``tensors`` (a saved pack).

    ``targets``: regex patterns or explicit module names. Returns the adapted module names.
    """
    key = _key(name)
    if name in attached_packs(model):
        raise ValueError(f"pack {name!r} is already attached")
    if tensors is not None:
        modules = sorted({k.rsplit(".", 1)[0] for k in tensors})
    elif isinstance(targets, (list, tuple)) and targets and all(
            isinstance(t, str) and not any(c in t for c in "\\()|*+?[]") for t in targets):
        modules = list(targets)
    else:
        modules = target_modules(model, targets)
    for module_name in modules:
        module = _module(model, module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{module_name} is not a Linear")
        weight = module.parametrizations.weight.original if parametrize.is_parametrized(module, "weight") \
            else module.weight
        out_f, in_f = weight.shape
        if tensors is None:
            factors = LowRankFactors(out_f, in_f, rank, alpha, device=weight.device, dtype=weight.dtype)
        else:
            A, B = tensors[f"{module_name}.A"], tensors[f"{module_name}.B"]
            if A.shape[1] != in_f or B.shape[0] != out_f or A.shape[0] != B.shape[1]:
                raise ValueError(f"{module_name}: pack factors {tuple(B.shape)}x{tuple(A.shape)} do not fit "
                                 f"the weight {tuple(weight.shape)}")
            factors = LowRankFactors(out_f, in_f, A.shape[0], alpha, device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                factors.A.copy_(A)
                factors.B.copy_(B)
        factors.requires_grad_(trainable)
        delta = _ensure_delta(module)
        delta.merged = None
        delta.packs[key] = factors
    return modules


def remove_pack(model: nn.Module, name: str, *, merge: bool = False) -> int:
    """Detach pack ``name``; a weight with no pack left gets its original base tensor back.

    ``merge=True`` instead folds the pack into the weights (``W += delta``, in place: the
    base changes) — used by an adapter-only sleep to produce an ordinary candidate model.
    Returns the number of weights the pack was attached to.
    """
    key = _key(name)
    count = 0
    for module in list(model.modules()):
        delta = _delta_of(module)
        if delta is None or key not in delta.packs:
            continue
        factors = delta.packs[key]
        if merge:
            original = module.parametrizations.weight.original
            with torch.no_grad():
                original.add_(factors.delta().to(original.dtype))
        del delta.packs[key]
        delta.merged = None
        count += 1
        if not len(delta.packs):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    if not count:
        raise KeyError(f"pack {name!r} is not attached")
    return count


def remove_all_packs(model: nn.Module) -> None:
    for name in attached_packs(model):
        remove_pack(model, name)


def freeze_base(model: nn.Module, *, packs=None) -> dict[str, bool]:
    """Only the given (default: all attached) packs' factors train; returns the previous flags."""
    wanted = None if packs is None else {_key(p) for p in packs}
    previous = {}
    for name, p in model.named_parameters():
        previous[name] = p.requires_grad
        adapter = ".parametrizations.weight.0.packs." in name
        key = name.split(".packs.", 1)[1].split(".", 1)[0] if adapter else None
        p.requires_grad_(adapter and (wanted is None or key in wanted))
    return previous


def adapter_parameter_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters() if ".parametrizations.weight.0.packs." in n]


@contextmanager
def packs_bypassed(model: nn.Module):
    """Inside the block every adapted weight reads the plain base weight: the base model's function
    without a second copy of it (a KL-to-base reference pass while a pack trains)."""
    deltas = [d for d in (_delta_of(m) for m in model.modules()) if d is not None]
    for delta in deltas:
        delta.bypass = True
    try:
        yield model
    finally:
        for delta in deltas:
            delta.bypass = False


def set_merged(model: nn.Module, merged: bool = True) -> int:
    """Cache ``W + delta`` per adapted weight for inference (False drops the caches)."""
    count = 0
    for module in model.modules():
        delta = _delta_of(module)
        if delta is None:
            continue
        delta.merged = None
        if merged:
            with torch.no_grad():
                original = module.parametrizations.weight.original
                value = original
                for factors in delta.packs.values():
                    value = value + factors.delta().to(original.dtype)
                delta.merged = value.detach()
        count += 1
    return count


# ----------------------------------------------------------------------------- packs on disk

@dataclass
class SkillPack:
    manifest: dict
    tensors: dict[str, torch.Tensor] = field(repr=False)
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def modules(self) -> list[str]:
        return sorted({k.rsplit(".", 1)[0] for k in self.tensors})


def _tensors_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().float().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pack_tensors(model: nn.Module, name: str) -> dict[str, torch.Tensor]:
    key = _key(name)
    out = {}
    for module_name, module in model.named_modules():
        delta = _delta_of(module)
        if delta is not None and key in delta.packs:
            factors = delta.packs[key]
            out[f"{module_name}.A"] = factors.A.detach().cpu().float().clone()
            out[f"{module_name}.B"] = factors.B.detach().cpu().float().clone()
    if not out:
        raise KeyError(f"pack {name!r} is not attached")
    return out


def make_pack(model: nn.Module, name: str, *, version: str = "1", description: str = "", training: dict | None = None,
              metrics: dict | None = None, lineage: list | None = None) -> SkillPack:
    """A ``SkillPack`` of the attached pack ``name`` bound to ``model``'s base."""
    tensors = pack_tensors(model, name)
    first = next(_delta_of(m).packs[_key(name)] for m in model.modules()
                 if _delta_of(m) is not None and _key(name) in _delta_of(m).packs)
    base_params = sum(v.numel() for v in base_state_dict(model).values() if v.is_floating_point())
    manifest = {
        "schema": SKILL_SCHEMA, "name": name, "version": str(version), "kind": "skill", "method": "lora",
        "description": description, "rank": first.rank, "alpha": first.alpha, "modules": sorted(
            {k.rsplit(".", 1)[0] for k in tensors}),
        "parameters": sum(v.numel() for v in tensors.values()),
        "base": {"fingerprint": base_fingerprint(model), "recipe": _model_recipe(model),
                 "parameters": base_params},
        "tensors_sha256": _tensors_digest(tensors), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training": training or {}, "metrics": metrics or {}, "lineage": lineage or [],
    }
    return SkillPack(manifest, tensors)


def save_pack(pack: SkillPack, directory: str | Path) -> Path:
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(pack.tensors, directory / "adapter.pt")
    (directory / "manifest.json").write_text(json.dumps(pack.manifest, indent=1), encoding="utf-8")
    pack.path = directory
    return directory


def load_pack(directory: str | Path) -> SkillPack:
    directory = Path(directory).expanduser()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SKILL_SCHEMA:
        raise ValueError(f"{directory} is not a {SKILL_SCHEMA} pack")
    tensors = torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True)
    if _tensors_digest(tensors) != manifest.get("tensors_sha256"):
        raise ValueError(f"{directory}: adapter tensors do not match the manifest hash")
    return SkillPack(manifest, tensors, directory)


def attach(model: nn.Module, pack: SkillPack | str | Path, *, check_base: bool = True, trainable: bool = False,
           name: str | None = None) -> str:
    """Attach a saved pack; refuses a pack trained on a different base unless ``check_base=False``."""
    if not isinstance(pack, SkillPack):
        pack = load_pack(pack)
    if check_base:
        expected = pack.manifest.get("base", {}).get("fingerprint")
        actual = base_fingerprint(model)
        if expected != actual:
            raise ValueError(f"skill pack {pack.name!r} was trained on base {str(expected)[:16]}, this model is "
                             f"{actual[:16]} (grow the pack with efficiency.lora.grow_pack, or pass check_base=False)")
    name = name or pack.name
    add_pack(model, name, alpha=float(pack.manifest["alpha"]), tensors=pack.tensors, trainable=trainable)
    return name


# ----------------------------------------------------------------------------- growth

def grow_pack(pack: SkillPack, source, grown, plan) -> SkillPack:
    """Map a pack of ``source`` onto ``grown`` (``efficiency.growth.grow_model`` output) exactly.

    With the grown weight W' = T W S (T: output replication and any rescale, S: the input
    split, summing replicas back to one), B' = T B and A' = A S give B'A' = T (B A) S: the
    grown base plus the grown pack computes what the source plus the pack computed.
    """
    from .growth import AxisSpec, _grow_tensor, axis_spec

    if pack.manifest.get("base", {}).get("fingerprint") not in (None, base_fingerprint(source)):
        raise ValueError("the pack was not trained on this source model")
    new_index = dict(enumerate(plan.layer_map))          # old core layer -> its grown index
    tensors = {}
    for key, value in pack.tensors.items():
        module_name, which = key.rsplit(".", 1)
        weight_name = f"{module_name}.weight"
        spec = axis_spec(weight_name)
        match = re.match(r"fast\.blocks\.(\d+)\.(.*)$", module_name)
        new_module = f"fast.blocks.{new_index[int(match.group(1))]}.{match.group(2)}" if match else module_name
        if which == "B":         # [out, r]: the weight's output side, with its rescaling
            part = AxisSpec((spec.dims[0], None), query=spec.query, depth=spec.depth) if spec.dims else spec
        else:                    # [r, in]: the weight's input side
            part = AxisSpec((None, spec.dims[1])) if spec.dims else spec
        tensors[f"{new_module}.{which}"] = _grow_tensor(value, weight_name, part, plan, label=f"pack:{key}")
    manifest = json.loads(json.dumps(pack.manifest))
    manifest["modules"] = sorted({k.rsplit(".", 1)[0] for k in tensors})
    manifest["parameters"] = sum(v.numel() for v in tensors.values())
    manifest["base"] = {"fingerprint": base_fingerprint(grown), "recipe": _model_recipe(grown),
                        "parameters": sum(v.numel() for v in base_state_dict(grown).values() if v.is_floating_point())}
    manifest["tensors_sha256"] = _tensors_digest(tensors)
    manifest["lineage"] = [*manifest.get("lineage", []),
                           {"op": "grow", "from_base": pack.manifest.get("base", {}).get("fingerprint"),
                            "growth": plan.describe(), "unix": time.time()}]
    return SkillPack(manifest, tensors)


__all__ = ["DEFAULT_TARGETS", "LowRankDelta", "LowRankFactors", "SKILL_SCHEMA", "SkillPack", "add_pack",
           "adapter_parameter_names", "attach", "attached_packs", "base_fingerprint", "base_state_dict",
           "freeze_base", "grow_pack", "load_pack", "make_pack", "pack_tensors", "packs_bypassed",
           "remove_all_packs", "remove_pack", "save_pack", "set_merged", "target_modules"]
