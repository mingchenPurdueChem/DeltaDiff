"""
MLCG-backed drop-in replacement for the paper's TorchMDCGGuidance.

Same interface, same "F_ext = F_mut - F_wt" mutation-guidance contract, same
pull-back-to-noise-space usage from src/models/score/r3.py -- only the force
backend differs: instead of a torchmd-net checkpoint, this evaluates the
pretrained MLCG (ClementiGroup/mlcg) CGSchNet+priors SumOut model twice per
step (once with wild-type residue identities, once with the mutant's) on the
SAME coordinates x_hat, and returns the force difference.

See scripts/mlcg_guidance.py for the standalone version of this mechanism
(load_mlcg_model / densify_sparse_buffers / MLCGGuidance) that this module
builds on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Reuse the already-validated MLCG loading/force utilities.
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from mlcg_guidance import load_mlcg_model  # noqa: E402
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY  # noqa: E402
from torch_geometric.data.collate import collate  # noqa: E402

from src.common.all_atom import compute_backbone  # noqa: E402
from src.common import residue_constants as rc  # noqa: E402


# Str2Str's local aatype order is AlphaFold/OpenFold style:
# A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
AF2_IDX_TO_1 = {
    0: "A", 1: "R", 2: "N", 3: "D", 4: "C",
    5: "Q", 6: "E", 7: "G", 8: "H", 9: "I",
    10: "L", 11: "K", 12: "M", 13: "F", 14: "P",
    15: "S", 16: "T", 17: "W", 18: "Y", 19: "V", 20: "X",
}
AA1_TO_AF2 = {v: k for k, v in AF2_IDX_TO_1.items() if v != "X"}

AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# mlcg's canonical CA-resolution embedding map
# (mlcg-tk/src/mlcg_tk/input_generator/embedding_maps.py: embedding_map_fivebead, capped <=20)
AA3_TO_MLCG = {
    "ALA": 1, "CYS": 2, "ASP": 3, "GLU": 4, "PHE": 5,
    "GLY": 6, "HIS": 7, "ILE": 8, "LYS": 9, "LEU": 10,
    "MET": 11, "ASN": 12, "PRO": 13, "GLN": 14, "ARG": 15,
    "SER": 16, "THR": 17, "VAL": 18, "TRP": 19, "TYR": 20,
}

# mlcg's 5-bead (backbone N/CA/CB/C/O) embedding map -- same table, full
# range (mlcg_tk.input_generator.embedding_maps.embedding_map_fivebead).
# Per-residue bead order is [N, CA, CB, C, O], except GLY which has no CB
# and instead encodes its identity directly on the CA bead: [N, CA(=6), C, O].
MLCG_N, MLCG_CA, MLCG_C, MLCG_O = 21, 22, 23, 24
GLY_MLCG_TYPE = AA3_TO_MLCG["GLY"]  # 6

# Str2Str atom37 backbone indices (src/common/residue_constants.py)
S2S_ATOM37_N = rc.atom_order["N"]
S2S_ATOM37_CA = rc.atom_order["CA"]
S2S_ATOM37_C = rc.atom_order["C"]
S2S_ATOM37_CB = rc.atom_order["CB"]
S2S_ATOM37_O = rc.atom_order["O"]
GLY_AF2_IDX = AA1_TO_AF2["G"]


def af2_aatype_to_mlcg_type(aatype: torch.Tensor) -> torch.Tensor:
    """Convert local AF2/OpenFold-style aatype tensor [B, L] to mlcg CA
    embedding-map atom types [B, L]."""
    if aatype.ndim != 2:
        raise ValueError(f"Expected aatype shape [B, L], got {tuple(aatype.shape)}")

    out = torch.empty_like(aatype, dtype=torch.long)
    for af2_idx, aa1 in AF2_IDX_TO_1.items():
        mask = aatype == af2_idx
        if not mask.any():
            continue
        if aa1 == "X":
            raise ValueError("Found unknown residue type X in aatype; mlcg guidance needs standard residues.")
        out[mask] = AA3_TO_MLCG[AA1_TO_3[aa1]]
    return out


class MLCGCGGuidance:
    """
    Wrap the pretrained MLCG (CGSchNet + priors) model as a differentiable
    mutation-guidance force:
        F_ext = F_mut - F_wt
    where F = -dE/dx for the same coordinates x but different residue
    (atom_type) embeddings -- DeltaDiff Eq. 7's Delta U_mut-wt, differentiated.
    """

    def __init__(
        self,
        checkpoint: str,
        template: str,
        template_mut: Optional[str] = None,
        mutations: Optional[List[Dict]] = None,
        xhat_is_nm: bool = True,
        use_mutant_minus_wt: bool = True,
        score_clip: float = 50.0,
        guidance_scale: float = 1.0,
        resolution: str = "ca",
        force_t_mid: float = 0.50,
        force_beta: float = 12.0,
        bead_order: Optional[List[str]] = None,
    ):
        self.checkpoint = checkpoint
        self.template_path = template
        # Only needed when a mutation changes a residue's bead COUNT, not
        # just its identity -- i.e. any mutation into or out of glycine.
        # GLY has no CB bead in this 5-bead scheme (4 beads vs 5 for every
        # other residue), so a WT<->GLY mutation genuinely changes the
        # topology (neighbor list, atom count), not just an atom-type label
        # on an unchanged skeleton. Reusing the WT template's connectivity
        # for the mutant side in that case would silently feed the model an
        # out-of-distribution "5-bead glycine with a synthetic CB" it was
        # never trained on. When None, WT and mutant share one template
        # (valid whenever neither side of the mutation is glycine).
        self.template_mut_path = template_mut
        self.mutations = mutations or []
        self.xhat_is_nm = xhat_is_nm
        self.use_mutant_minus_wt = use_mutant_minus_wt
        self.score_clip = float(score_clip)
        self.guidance_scale = float(guidance_scale)
        assert resolution in ("ca", "5bead"), resolution
        self.resolution = resolution
        self.force_t_mid = float(force_t_mid)
        self.force_beta = float(force_beta)
        # per-residue backbone bead order for non-GLY residues; must match
        # whatever order the guidance checkpoint's own training template
        # actually uses (varies per dataset -- see _backbone_5bead).
        self.bead_order = list(bead_order) if bead_order is not None else ["N", "CA", "C", "O", "CB"]
        self._bead_order_verified = False
        self._bead_order_verified_mut = False

        self.model = None
        self.model_device = None
        self._template_batch = None
        self._template_single = None
        self._batch_topology_cache = {}  # B -> collated batch-of-B topology (pos/atom_types get overwritten per call)
        self._template_batch_mut = None
        self._template_single_mut = None
        self._batch_topology_cache_mut = {}
        self._residue_is_gly_wt = None  # [L] bool, only used for resolution="5bead"
        self._residue_is_gly_mut = None

    def _ensure_model(self, device: torch.device):
        if self.model is not None and self.model_device == device:
            return
        self.model = load_mlcg_model(self.checkpoint, device=str(device))
        # We only ever need d(energy)/d(coordinates) -- never d(.)/d(theta)
        # (inference only, no training). Freezing lets autograd skip saving
        # activations that exist solely to support parameter gradients
        # (e.g. Linear's backward doesn't need to keep the input around for
        # dL/dinput, only for dL/dweight), which is a large chunk of the
        # memory this model's own internal energy->force autograd.grad call
        # otherwise retains.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model_device = device

        structs = torch.load(self.template_path, map_location="cpu", weights_only=False)
        template = structs[0] if isinstance(structs, (list, tuple)) else structs
        batch, _, _ = collate(template.__class__, data_list=[template], increment=True, add_batch=True)
        self._template_batch = batch.to(device)
        # Keep the raw (uncollated) single-structure template too, so the
        # per-replica force loop can build ONE properly batched graph of B
        # replicas via collate() instead of looping model(single_structure)
        # B times -- looping was the actual bottleneck (an unbatched GNN
        # forward+backward per replica, per WT/mutant, per timestep).
        self._template_single = template.to(device)

        if self.template_mut_path is not None:
            structs_mut = torch.load(self.template_mut_path, map_location="cpu", weights_only=False)
            template_mut = structs_mut[0] if isinstance(structs_mut, (list, tuple)) else structs_mut
            batch_mut, _, _ = collate(
                template_mut.__class__, data_list=[template_mut], increment=True, add_batch=True
            )
            self._template_batch_mut = batch_mut.to(device)
            self._template_single_mut = template_mut.to(device)

    def build_mutant_aatype(
        self,
        wt_aatype: torch.Tensor,
        residue_index: Optional[torch.Tensor] = None,
        chain_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Supported mutation specs (same as TorchMDCGGuidance):
          - {"seq_idx": 42, "wt": "D", "mut": "N"}
          - {"resid": 162, "wt": "D", "mut": "N"}
          - {"resid": 162, "chain_index": 0, "wt": "D", "mut": "N"}
        """
        mut_aatype = wt_aatype.clone()

        for spec in self.mutations:
            if "seq_idx" in spec:
                seq_idx = int(spec["seq_idx"])
                if seq_idx < 0 or seq_idx >= wt_aatype.shape[1]:
                    raise IndexError(f"seq_idx={seq_idx} out of range for length {wt_aatype.shape[1]}")
                mask = torch.zeros_like(wt_aatype, dtype=torch.bool)
                mask[:, seq_idx] = True
            elif "resid" in spec:
                if residue_index is None:
                    raise ValueError("Mutation spec uses 'resid' but residue_index was not provided.")
                mask = residue_index == int(spec["resid"])
                if "chain_index" in spec:
                    if chain_index is None:
                        raise ValueError("Mutation spec uses 'chain_index' but chain_index was not provided.")
                    mask = mask & (chain_index == int(spec["chain_index"]))
            else:
                raise ValueError(f"Mutation spec must contain either 'seq_idx' or 'resid'. Got: {spec}")

            if not mask.any():
                raise ValueError(f"Mutation spec matched no residues: {spec}")

            wt_expected = AA1_TO_AF2[spec["wt"]]
            mut_target = AA1_TO_AF2[spec["mut"]]

            observed = wt_aatype[mask]
            if not torch.all(observed == wt_expected):
                obs_unique = torch.unique(observed).tolist()
                raise ValueError(
                    f"WT residue mismatch for spec {spec}. Observed AF2 indices {obs_unique}, expected {wt_expected}."
                )

            mut_aatype[mask] = mut_target

        return mut_aatype

    # ------------------------------------------------------------------
    # 5-bead (N, CA, CB, C, O) backbone reconstruction + per-residue sum
    # ------------------------------------------------------------------

    def _ensure_residue_layout(self, wt_aatype: torch.Tensor, mut_aatype: torch.Tensor):
        """GLY has no CB bead in the 5-bead scheme (its CA bead carries the
        GLY identity directly instead). Cache which residues are GLY,
        SEPARATELY for the wild-type and mutant sequences -- a mutation into
        or out of glycine genuinely changes that residue's bead count (5 ->
        4 or 4 -> 5), not just its identity label, so the two masks can
        differ at the mutated position.
        """
        if self._residue_is_gly_wt is None:
            self._residue_is_gly_wt = (wt_aatype[0] == GLY_AF2_IDX).detach().cpu()
        if self._residue_is_gly_mut is None:
            self._residue_is_gly_mut = (mut_aatype[0] == GLY_AF2_IDX).detach().cpu()

    def _backbone_5bead(
        self, rigids_hat, psi_hat: torch.Tensor, aatype: torch.Tensor, is_gly_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct real-Angstrom N/CA/CB/C/O positions (via Str2Str's
        own compute_backbone -- the exact function it uses to write out
        final PDBs) and flatten them into mlcg's 5-bead bead order.

        is_gly_mask: [L] bool, the Gly-ness layout to use for THIS aatype
        (wt or mut -- callers must pass the mask matching whichever aatype
        is given, since the two can differ at a mutated Gly position).

        Returns
        -------
        pos: [B, n_atoms, 3] flattened bead positions
        types: [B, n_atoms] mlcg embedding-map atom types for this aatype
        residue_of_atom: [n_atoms] which residue (0..L-1) each bead belongs to
        """
        atom37, _, _, _ = compute_backbone(rigids_hat, psi_hat, aatype=aatype)
        # atom37: [B, L, 37, 3], real Angstrom (same convention as the final
        # PDB output -- see diffusion_module.py's own compute_backbone call).
        B, L = atom37.shape[0], atom37.shape[1]
        device = atom37.device

        pos_chunks, type_chunks, residue_of_atom = [], [], []
        for i in range(L):
            is_gly = bool(is_gly_mask[i])
            n_pos = atom37[:, i, S2S_ATOM37_N]
            ca_pos = atom37[:, i, S2S_ATOM37_CA]
            c_pos = atom37[:, i, S2S_ATOM37_C]
            o_pos = atom37[:, i, S2S_ATOM37_O]

            if is_gly:
                pos_chunks += [n_pos, ca_pos, c_pos, o_pos]
                types_i = [MLCG_N, GLY_MLCG_TYPE, MLCG_C, MLCG_O]
                residue_of_atom += [i] * 4
            else:
                cb_pos = atom37[:, i, S2S_ATOM37_CB]
                cb_type = af2_aatype_to_mlcg_type(aatype[:, i : i + 1])[:, 0]  # [B]
                # Bead order within a residue is NOT a fixed universal mlcg
                # convention -- mlcg-tk's gen_sim_input preserves whatever
                # order the *source PDB* lists each atom name in, so it
                # varies per dataset (observed: [N,CA,CB,C,O] for the
                # bundled chignolin fixture, [N,CA,C,O,CB] for an NMR-derived
                # BBL structure, [N,CA,C,CB,O] for an ESMFold-derived
                # chignolin structure). self.bead_order pins the exact
                # sequence to match -- verify it against the actual
                # template's atom_types before trusting a run.
                slots = {
                    "N": (n_pos, torch.full((B,), MLCG_N, dtype=torch.long, device=device)),
                    "CA": (ca_pos, torch.full((B,), MLCG_CA, dtype=torch.long, device=device)),
                    "C": (c_pos, torch.full((B,), MLCG_C, dtype=torch.long, device=device)),
                    "O": (o_pos, torch.full((B,), MLCG_O, dtype=torch.long, device=device)),
                    "CB": (cb_pos, cb_type),
                }
                types_i = []
                for name in self.bead_order:
                    p, t = slots[name]
                    pos_chunks.append(p)
                    types_i.append(t)
                type_chunks.append(torch.stack(types_i, dim=1))  # [B, 5]
                residue_of_atom += [i] * 5
                continue

            type_chunks.append(
                torch.tensor(types_i, dtype=torch.long, device=device)
                .unsqueeze(0).expand(B, -1)
            )

        pos = torch.stack(pos_chunks, dim=1)  # [B, n_atoms, 3]
        types = torch.cat(type_chunks, dim=1)  # [B, n_atoms]
        residue_of_atom = torch.tensor(residue_of_atom, dtype=torch.long, device=device)
        return pos, types, residue_of_atom

    def _get_batch_topology(self, B: int, which: str = "wt"):
        """Returns a batched graph of B replicas of the fixed template
        topology (neighbor lists etc.), built via collate() ONCE per unique
        B and cached thereafter. Same-sequence-variant calls (all WT calls,
        or all mutant calls when a separate template_mut is in play) share
        the exact same connectivity -- only pos/atom_types ever change -- so
        rebuilding the whole nested batch structure from scratch on every
        call (as the first version of this batching fix did) is pure waste:
        collate() and its underlying per-item .clone()/recursive_apply are
        real Python-level overhead that dominates when repeated 2000+ times
        over a sampling run.

        which: "wt" or "mut" -- selects which template/cache to use. Only
        matters when template_mut was provided (a Gly-transition mutation);
        otherwise both route to the same WT template.
        """
        use_mut = which == "mut" and self._template_single_mut is not None
        cache = self._batch_topology_cache_mut if use_mut else self._batch_topology_cache
        template_single = self._template_single_mut if use_mut else self._template_single

        cached = cache.get(B)
        if cached is not None:
            return cached
        data_list = [template_single.clone() for _ in range(B)]
        batched, _, _ = collate(
            template_single.__class__, data_list=data_list, increment=True, add_batch=True
        )
        batched = batched.to(self.model_device)
        cache[B] = batched
        return batched

    def _run_batched_model(self, pos: torch.Tensor, types: torch.Tensor, which: str = "wt") -> torch.Tensor:
        """pos/types: [B, n_atoms, (3|)]. Returns per-atom force [B, n_atoms, 3].

        Runs ONE batched forward+backward pass of B replicas through the
        model, reusing a cached batch topology (see _get_batch_topology) and
        only overwriting pos/atom_types -- plain flat tensor writes, not the
        Python-level per-item rebuild collate() would otherwise redo every
        call. The original per-replica-loop version called
        `self.model(single)` B times in Python (once per replica); at
        replica_per_batch=64, 1000 timesteps, x2 for WT/mutant, that's
        ~128,000 tiny sequential model calls instead of ~2,000 batched ones
        -- the actual cause of multi-hour runtimes on the 5-bead model (small
        enough to go unnoticed on the lighter Ca-only case).

        which: "wt" or "mut" -- routes to the matching template (see
        _get_batch_topology); required whenever WT/mutant differ in atom
        count (a Gly-transition mutation).
        """
        use_mut = which == "mut" and self._template_single_mut is not None
        template_single = self._template_single_mut if use_mut else self._template_single

        B, n_atoms = pos.shape[0], pos.shape[1]
        model_dtype = next(self.model.parameters()).dtype
        assert template_single.pos.shape[0] == n_atoms, (
            f"{'mutant' if use_mut else 'WT'} template has {template_single.pos.shape[0]} atoms, "
            f"reconstructed backbone has {n_atoms} -- layout mismatch."
        )
        batched = self._get_batch_topology(B, which=which)
        with torch.enable_grad():
            batched.pos = (
                pos.reshape(B * n_atoms, 3).to(device=self.model_device, dtype=model_dtype)
                .detach().clone().requires_grad_(True)
            )
            batched.atom_types = types.reshape(B * n_atoms).to(self.model_device)
            out = self.model(batched)
        forces_flat = out.out[FORCE_KEY].to(model_dtype)  # [B*n_atoms, 3]
        return forces_flat.view(B, n_atoms, 3)

    def _batched_force_5bead(
        self, pos: torch.Tensor, types: torch.Tensor, which: str = "wt"
    ) -> torch.Tensor:
        """pos/types: [B, n_atoms, (3|)]. Returns per-atom force [B, n_atoms, 3]
        via the 5-bead mlcg model (same mechanism as _batched_force, just
        without relying on a single fixed template's atom_types -- we swap
        in the atom_types matching this call's aatype directly)."""
        return self._run_batched_model(pos, types, which=which)

    @staticmethod
    def _sum_per_residue(atom_forces: torch.Tensor, residue_of_atom: torch.Tensor, n_residues: int) -> torch.Tensor:
        """atom_forces: [B, n_atoms, 3] -> [B, n_residues, 3].

        Translating a residue's rigid-frame origin (its CA translation DOF)
        shifts every backbone atom rigidly attached to that frame (N, CA,
        CB, C, O) by the exact same displacement -- so d(energy)/d(CA
        translation) is exactly the sum of the per-atom forces on that
        residue's beads. This is what makes "sum N/CA/C(/O/CB) forces onto
        CA" the mathematically correct way to use a higher-resolution
        potential to guide a CA-only translational diffuser.
        """
        B = atom_forces.shape[0]
        out = torch.zeros(B, n_residues, 3, dtype=atom_forces.dtype, device=atom_forces.device)
        out.index_add_(1, residue_of_atom, atom_forces)
        return out

    def _compute_force_5bead(
        self,
        rigids_hat,
        psi_hat: torch.Tensor,
        wt_aatype: torch.Tensor,
        mut_aatype: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_residue_layout(wt_aatype, mut_aatype)
        L = wt_aatype.shape[1]
        has_mut_template = self._template_single_mut is not None

        pos_wt, types_wt, res_of_atom_wt = self._backbone_5bead(
            rigids_hat, psi_hat, wt_aatype, is_gly_mask=self._residue_is_gly_wt
        )
        pos_mut, types_mut, res_of_atom_mut = self._backbone_5bead(
            rigids_hat, psi_hat, mut_aatype, is_gly_mask=self._residue_is_gly_mut
        )
        # WT and mutant atom counts only match when neither side of the
        # mutation is glycine; a Gly-transition mutation (needs template_mut)
        # legitimately produces different-length reconstructions.
        if not has_mut_template:
            assert torch.equal(res_of_atom_wt, res_of_atom_mut), (
                "WT/mutant atom-count mismatch but no template_mut was provided -- "
                "this mutation changes a residue's bead count (glycine transition); "
                "pass guidance.template_mut pointing at a properly-built mutant topology."
            )

        if not self._bead_order_verified:
            # types_wt encodes the WT sequence's atom-type pattern exactly
            # like the training template does; if bead_order is wrong for
            # this checkpoint's dataset, this mismatches immediately rather
            # than silently feeding scrambled atom identities to the model.
            template_types = self._template_batch.atom_types
            if not torch.equal(types_wt[0], template_types):
                raise ValueError(
                    f"bead_order={self.bead_order} does not match the guidance "
                    f"template's own atom_types. Reconstructed: {types_wt[0].tolist()} "
                    f"vs template: {template_types.tolist()}. Check the source PDB's "
                    f"per-residue atom order used to build the template."
                )
            self._bead_order_verified = True

        if has_mut_template and not self._bead_order_verified_mut:
            template_types_mut = self._template_batch_mut.atom_types
            if not torch.equal(types_mut[0], template_types_mut):
                raise ValueError(
                    f"bead_order={self.bead_order} does not match the guidance "
                    f"template_mut's own atom_types. Reconstructed: {types_mut[0].tolist()} "
                    f"vs template_mut: {template_types_mut.tolist()}. Check the mutant "
                    f"topology's own per-residue atom order."
                )
            self._bead_order_verified_mut = True

        F_wt_atoms = self._batched_force_5bead(pos_wt, types_wt, which="wt")
        F_mut_atoms = self._batched_force_5bead(pos_mut, types_mut, which="mut" if has_mut_template else "wt")

        F_wt = self._sum_per_residue(F_wt_atoms, res_of_atom_wt, L)
        F_mut = self._sum_per_residue(F_mut_atoms, res_of_atom_mut, L)

        F_ext = (F_mut - F_wt) if self.use_mutant_minus_wt else F_mut
        return F_ext

    def _batched_force(self, pos_angstrom: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """pos_angstrom: [B, L, 3]; atom_types: [B, L] (mlcg embedding ints).
        Returns [B, L, 3] force = -dE/dpos, via autograd through the mlcg model.
        """
        return self._run_batched_model(pos_angstrom, atom_types)

    def compute_force(
        self,
        x_hat: torch.Tensor,
        wt_aatype: torch.Tensor,
        mut_aatype: torch.Tensor,
        residue_mask: Optional[torch.Tensor] = None,
        rigids_hat=None,
        psi_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns force in the SAME coordinate units as x_hat (numeric
        values only -- the caller is responsible for detaching/pulling this
        back into the noise-space score, see r3.py's reverse())."""
        self._ensure_model(x_hat.device)

        if self.resolution == "5bead":
            if rigids_hat is None or psi_hat is None:
                raise ValueError(
                    "resolution='5bead' requires rigids_hat/psi_hat (the network's "
                    "own x0 rotation+translation estimate and psi torsion) to "
                    "reconstruct N/CA/CB/C/O backbone atoms -- got None. Check that "
                    "diffusion_module.py/frame.py are passing them through to "
                    "R3Diffuser.reverse()."
                )
            F_ext = self._compute_force_5bead(rigids_hat, psi_hat, wt_aatype, mut_aatype)
            # F_ext is real-Angstrom (compute_backbone's own convention); if
            # x_hat/x_t are in mlcg's "scaled" (nm-ish) units, convert to
            # match, exactly as the CA-only path does below.
            if self.xhat_is_nm:
                F_ext = F_ext * 10.0
            return F_ext.to(dtype=x_hat.dtype)

        if self.xhat_is_nm:
            pos_angstrom = x_hat.detach() * 10.0
            force_backscale = 10.0
        else:
            pos_angstrom = x_hat.detach()
            force_backscale = 1.0

        z_wt = af2_aatype_to_mlcg_type(wt_aatype)
        z_mut = af2_aatype_to_mlcg_type(mut_aatype)

        F_wt = self._batched_force(pos_angstrom, z_wt)
        F_mut = self._batched_force(pos_angstrom, z_mut)

        F_ext = (F_mut - F_wt) if self.use_mutant_minus_wt else F_mut
        F_ext = F_ext * force_backscale
        return F_ext.to(dtype=x_hat.dtype)
