"""
MLCG-guided sampling utilities for Str2Str.

Wires a pretrained MLCG (machine-learned coarse-grained) energy model
- Charron, N.E. et al. "Navigating protein landscapes with a machine-learned
  transferable coarse-grained model", Nature Chemistry (2025)
- library: https://github.com/ClementiGroup/mlcg
as a physical guidance potential for the reverse SDE of Str2Str
(https://github.com/lujiarui/Str2Str, Lu et al., ICLR 2024), following the
guided-inference scheme of DeltaDiff (Eq. 5-8):

    dU_t(x(t)) ~= U_mut-wt(xhat(0))                                  (Eq. 7)
    dx = [f(x,t) - g(t)^2 (s_theta(x(t),t) + lambda(t) * F(x(t),t))] dt
         + g(t) dw                                                    (Eq. 8)

where F(x(t), t) = grad_x [-U(xhat(0))] is the physical force from the MLCG
potential evaluated at the denoised estimate xhat(0), and lambda(t) is an
(empirical) time-dependent guidance strength.

This module only implements the *guidance* half (loading an MLCG model and
turning arbitrary CA coordinates into a force via autograd). It is
architecture-agnostic with respect to the diffusion backbone: pass it
whatever xhat(0) Cartesian coordinates your SDE step produces (for Str2Str,
this is the translation component of the predicted frames).

Known checkpoint quirk (see README section below): priors merged with
`mlcg-combine_model` from the tests fixtures store their lookup-table
buffers (x0/k, k1s/k2s/v_0, sigma, alpha, r_0, ...) as sparse COO tensors.
torch>=2.x's `aten::index.Tensor` has no sparse-COO kernel, so any prior
class that indexes those buffers with `buf[interaction_types]` crashes.
`densify_sparse_buffers` below is a load-time, architecture-agnostic fix:
it walks every buffer in the model and converts sparse ones to dense
in-place. (mlcg's own `Harmonic`/`FourierSeries` prior classes were also
patched in the editable install at repos/mlcg/src/mlcg/nn/prior/ to
densify defensively, but `densify_sparse_buffers` alone is sufficient and
covers `Repulsion` and any other prior type without further source edits.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.data.atomic_data import AtomicData
from torch_geometric.data.collate import collate


def _install_legacy_pyg_shim() -> None:
    """Older (~2023) mlcg checkpoints were pickled against a torch_geometric
    version where `Inspector` lived at
    `torch_geometric.nn.conv.utils.inspector`. It has since moved to
    `torch_geometric.inspector`. Aliasing the old module path lets
    `torch.load` unpickle those checkpoints under modern torch_geometric.
    Note: this resolves the *module lookup*; it does not guarantee the
    `Inspector` object's internal attributes are still compatible (they
    were not, as of mlcg 0.1.4 / torch_geometric 2.8 vs. the 2023-era
    Zenodo `model_and_prior.pt` release for the transferable model) so a
    full nn.Module pickled by torch_geometric's MessagePassing may still
    fail deeper in `__setstate__`. It is applied unconditionally here
    because it is a harmless no-op for checkpoints that don't need it.
    """
    import torch_geometric.inspector as _inspector

    sys.modules.setdefault(
        "torch_geometric.nn.conv.utils.inspector", _inspector
    )


class _safe_unpickle_message_passing:
    """Some checkpoints (e.g. the Zenodo `model_and_prior.pt` transferable
    model, pickled ~2023) have a `MessagePassing.__setstate__` that calls
    `self._set_jittable_templates()` during unpickling itself -- which
    dereferences `self.inspector._cls`, an attribute that doesn't exist on
    the stale pickled Inspector. This crashes *inside* `torch.load`, before
    the object is ever returned, so a post-load fix (fix_legacy_message_
    passing) never gets a chance to run.

    Context manager that temporarily replaces `__setstate__` with a plain
    `__dict__` restore (no jittable-template rebuild) for the duration of
    the `torch.load` call. Combine with `fix_legacy_message_passing(model)`
    afterward to properly rebuild the inspector/templates on the now fully
    -unpickled object.
    """

    def __enter__(self):
        from torch_geometric.nn import MessagePassing

        self._orig = MessagePassing.__setstate__

        def _safe_setstate(obj, data):
            obj.__dict__.update(data)

        MessagePassing.__setstate__ = _safe_setstate
        return self

    def __exit__(self, *exc):
        from torch_geometric.nn import MessagePassing

        MessagePassing.__setstate__ = self._orig
        return False


def fix_legacy_message_passing(model: torch.nn.Module) -> torch.nn.Module:
    """Some mlcg checkpoints (e.g. the bundled PaiNN chignolin fixture) were
    pickled against an older torch_geometric whose `MessagePassing` didn't
    yet have `decomposed_layers`/`explain` etc. Unpickling restores whatever
    was in the old `__dict__`, silently omitting attributes that didn't
    exist back then -- `propagate()` then hits a plain AttributeError.
    Backfill sane defaults (matching current MessagePassing.__init__) on
    every MessagePassing submodule that's missing them.
    """
    from collections import OrderedDict

    from torch_geometric.inspector import Inspector
    from torch_geometric.nn import MessagePassing

    # decomposed_layers/explain are properties backed by these private
    # fields; writing the public name into __dict__ is a no-op since a
    # data descriptor (the property) always wins over instance __dict__.
    scalar_defaults = {
        "_decomposed_layers": 1,
        "_explain": None,
        "_edge_mask": None,
        "_loop_mask": None,
        "_apply_sigmoid": True,
    }
    hook_dicts = [
        "_propagate_forward_pre_hooks", "_propagate_forward_hooks",
        "_message_forward_pre_hooks", "_message_forward_hooks",
        "_aggregate_forward_pre_hooks", "_aggregate_forward_hooks",
        "_message_and_aggregate_forward_pre_hooks", "_message_and_aggregate_forward_hooks",
        "_edge_update_forward_pre_hooks", "_edge_update_forward_hooks",
    ]

    for module in model.modules():
        if not isinstance(module, MessagePassing):
            continue
        for name, default in scalar_defaults.items():
            if name not in module.__dict__:
                module.__dict__[name] = default
        for name in hook_dicts:
            if name not in module.__dict__:
                module.__dict__[name] = OrderedDict()

        # `self.inspector` (and everything derived from it) may be a stale
        # object pickled against an older torch_geometric that lacks
        # attributes the current Inspector.get_signature()/propagate() need
        # (e.g. `_signature_dict`). Rebuild it fresh -- exactly mirroring
        # MessagePassing.__init__ -- instead of trying to patch the old one.
        needs_rebuild = (
            "inspector" not in module.__dict__
            or not hasattr(module.__dict__["inspector"], "_signature_dict")
        )
        if needs_rebuild:
            module.inspector = Inspector(module.__class__)
            module.inspector.inspect_signature(module.message)
            module.inspector.inspect_signature(module.aggregate, exclude=[0, "aggr"])
            module.inspector.inspect_signature(module.message_and_aggregate, [0])
            module.inspector.inspect_signature(module.update, exclude=[0])
            module.inspector.inspect_signature(module.edge_update)

            module._user_args = module.inspector.get_flat_param_names(
                ["message", "aggregate", "update"], exclude=module.special_args
            )
            module._fused_user_args = module.inspector.get_flat_param_names(
                ["message_and_aggregate", "update"], exclude=module.special_args
            )
            module._edge_user_args = module.inspector.get_param_names(
                "edge_update", exclude=module.special_args
            )
            module.fuse = module.inspector.implements("message_and_aggregate")
            if module.aggr is not None:
                from torch_geometric.nn.conv.message_passing import FUSE_AGGRS
                module.fuse &= isinstance(module.aggr, str) and module.aggr in FUSE_AGGRS

            module._set_jittable_templates()
    return model


def densify_sparse_buffers(model: torch.nn.Module) -> torch.nn.Module:
    """Convert any sparse-COO buffers in `model` to dense, in place.

    Works around checkpoints (produced by `mlcg-combine_model` from the
    library's own `tests/continuity` fixtures) whose prior lookup tables
    are stored as `torch.sparse_coo_tensor`. Fancy/advanced indexing
    (`buf[interaction_types]`, used by every prior class to select
    per-interaction parameters) is not implemented for sparse-COO tensors
    in torch>=2.x, on either CPU or CUDA.
    """
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_sparse:
                module._buffers[name] = buf.to_dense()
    return model


def load_mlcg_model(
    checkpoint_path: str | Path, device: str = "cuda"
) -> torch.nn.Module:
    """Load a `model_and_prior.pt` / `mlcg-combine_model` output and
    prepare it for use as a guidance potential (eval mode, dense buffers,
    on `device`)."""
    _install_legacy_pyg_shim()
    device = device if torch.cuda.is_available() else "cpu"
    with _safe_unpickle_message_passing():
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    densify_sparse_buffers(model)
    fix_legacy_message_passing(model)
    return model


class MLCGGuidance:
    """Turns Cartesian CG-bead coordinates into an energy/force via a
    pretrained MLCG model, for a *fixed* topology (sequence + neighbor
    list). Rebuild a new instance if the sequence changes.

    Parameters
    ----------
    model:
        A loaded `SumOut` energy model (see `load_mlcg_model`).
    template:
        An `AtomicData` instance with the right `atom_types` and
        `neighbor_list` for the molecule being sampled (positions are
        overwritten on every call). For a fixed-topology system, this can
        come straight from the `tests/continuity/model_ckpts/structures_*
        .pt` fixtures shipped with mlcg, or be built with `mlcg-tk` /
        `AtomicData.from_points` + a neighbor-list builder for a new
        sequence.
    """

    def __init__(self, model: torch.nn.Module, template: AtomicData):
        self.model = model
        self.device = next(model.parameters()).device
        batch, _, _ = collate(
            template.__class__, data_list=[template], increment=True, add_batch=True
        )
        self.template_batch = batch.to(self.device)

    @torch.no_grad()
    def _n_beads(self) -> int:
        return self.template_batch.pos.shape[0]

    def energy_force(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        """pos: [N_beads, 3] Cartesian coordinates (CPU or CUDA, any
        dtype/grad state). Returns (energy [1], force [N_beads, 3]) with
        force = -dE/dpos, computed via autograd through the MLCG model.
        Safe to call inside an outer autograd graph (e.g. from within a
        diffusion sampler step) -- gradients will flow back through `pos`.
        """
        assert pos.shape == self.template_batch.pos.shape, (
            f"expected {self.template_batch.pos.shape} CG beads, got {pos.shape}"
        )
        batch = self.template_batch.clone()
        with torch.enable_grad():
            batch.pos = pos.to(self.device).requires_grad_(True)
            out = self.model(batch)
        # SumOut pools per-submodel predictions into out.out[target]; the
        # top-level `out[FORCE_KEY]` (if present) is just ground-truth
        # reference data carried over from the AtomicData template, not
        # the model's prediction.
        return out.out[ENERGY_KEY], out.out[FORCE_KEY]


def guidance_force(guidance: MLCGGuidance, xhat0: Tensor) -> Tensor:
    """DeltaDiff Eq. 7 guidance force, F(x(t),t) = grad_x [-U(xhat(0))],
    i.e. simply the MLCG force evaluated at the denoised estimate."""
    _, force = guidance.energy_force(xhat0)
    return force


def guided_score(
    score: Tensor,
    xhat0: Tensor,
    guidance: MLCGGuidance,
    lam: float,
) -> Tensor:
    """DeltaDiff Eq. 8 guided score: s_theta(x(t),t) + lambda(t) * F(x(t),t).

    `score` is Str2Str's own predicted score (or, for the translation
    component of a frame-based model, the score restricted to
    translations). `xhat0` is the model's denoised estimate of the CG
    (e.g. CA-only) coordinates at this step -- for Str2Str's frame
    diffusion, this is `frames.trans` evaluated from the predicted
    x0-frames. `lam` is the empirical time-dependent guidance weight
    lambda(t) from Sec. 2.2 of the DeltaDiff paper.
    """
    return score + lam * guidance_force(guidance, xhat0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(
            Path(__file__).resolve().parents[1]
            / "mlcg_pretrained"
            / "chignolin_ca_combined.pt"
        ),
        help="Path to a model_and_prior.pt / mlcg-combine_model output.",
    )
    parser.add_argument(
        "--template",
        default=str(
            Path(__file__).resolve().parents[1]
            / "repos"
            / "mlcg"
            / "tests"
            / "continuity"
            / "model_ckpts"
            / "structures_cln_ca.pt"
        ),
        help="Path to an AtomicData list fixture defining the topology.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_mlcg_model(args.checkpoint, device=args.device)
    structs = torch.load(args.template, map_location="cpu", weights_only=False)
    template = structs[0] if isinstance(structs, (list, tuple)) else structs

    guidance = MLCGGuidance(model, template)

    torch.manual_seed(0)
    x = template.pos.clone() + 0.1 * torch.randn_like(template.pos)
    energy, force = guidance.energy_force(x)

    print(f"beads: {x.shape[0]}, device: {guidance.device}")
    print(f"energy: {energy.item():.4f}")
    print(f"|force| mean: {force.norm(dim=-1).mean().item():.4f}")
    print(f"force[:3]:\n{force[:3]}")

    # Sanity check: nudging positions along the force direction should
    # lower the energy (force = -dE/dx).
    with torch.no_grad():
        x = x.to(force.device)
        x_step = x + 1e-3 * force / force.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    energy_after, _ = guidance.energy_force(x_step)
    print(
        f"energy after small step along force: {energy_after.item():.4f} "
        f"({'lower' if energy_after.item() < energy.item() else 'HIGHER'} -- expected lower)"
    )
