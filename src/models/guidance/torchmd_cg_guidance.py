from __future__ import annotations

from typing import Dict, List, Optional

import torch


# Your model's local aatype order is AlphaFold/OpenFold style:
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

# Official embedding dictionary used in the paper tutorial.
# Do NOT replace with your local aatype integers.
AA3_TO_TORCHMD = {
    "ALA": 1,
    "GLY": 2,
    "PHE": 3,
    "TYR": 4,
    "ASP": 5,
    "GLU": 6,
    "TRP": 7,
    "PRO": 8,
    "ASN": 9,
    "GLN": 10,
    "HIS": 11,
    "SER": 12,
    "THR": 13,
    "VAL": 14,
    "MET": 15,
    "CYS": 16,
    "NLE": 17,
    "ARG": 19,
    "LYS": 20,
    "LEU": 21,
    "ILE": 22,
}


def af2_aatype_to_torchmd_z(aatype: torch.Tensor) -> torch.Tensor:
    """
    Convert local AF2/OpenFold-style aatype tensor [B, L] to TorchMD paper embeddings [B, L].
    """
    if aatype.ndim != 2:
        raise ValueError(f"Expected aatype shape [B, L], got {tuple(aatype.shape)}")

    device = aatype.device
    out = torch.empty_like(aatype, dtype=torch.long, device=device)

    for af2_idx, aa1 in AF2_IDX_TO_1.items():
        mask = (aatype == af2_idx)
        if not mask.any():
            continue
        if aa1 == "X":
            raise ValueError("Found unknown residue type X in aatype; paper checkpoint expects standard mapped residue types.")
        aa3 = AA1_TO_3[aa1]
        out[mask] = AA3_TO_TORCHMD[aa3]

    return out


def _pack_graph_batch(
    pos: torch.Tensor,          # [B, L, 3]
    z_full: torch.Tensor,       # [B, L]
    residue_mask: Optional[torch.Tensor] = None,   # [B, L]
):
    """
    Flatten batch of variable-length residue graphs into torchmd-net graph inputs.
    Returns:
      z_flat:    [Ntot]
      pos_flat:  [Ntot, 3]
      batch_idx: [Ntot]
      keep_list: list of boolean masks, one per batch row
    """
    if pos.ndim != 3 or z_full.ndim != 2:
        raise ValueError(f"Expected pos [B,L,3] and z_full [B,L], got {tuple(pos.shape)} and {tuple(z_full.shape)}")

    B, L, _ = pos.shape
    device = pos.device

    z_parts = []
    pos_parts = []
    batch_parts = []
    keep_list = []

    for b in range(B):
        if residue_mask is None:
            keep = torch.ones(L, dtype=torch.bool, device=device)
        else:
            keep = residue_mask[b].to(dtype=torch.bool)

        keep_list.append(keep)
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            raise ValueError(f"Batch row {b} has zero kept residues.")

        z_parts.append(z_full[b, keep])
        pos_parts.append(pos[b, keep])
        batch_parts.append(torch.full((n_keep,), b, dtype=torch.long, device=device))

    z_flat = torch.cat(z_parts, dim=0)
    pos_flat = torch.cat(pos_parts, dim=0)
    batch_idx = torch.cat(batch_parts, dim=0)

    return z_flat, pos_flat, batch_idx, keep_list


def _unpack_forces(
    flat_forces: torch.Tensor,      # [Ntot, 3]
    keep_list: List[torch.Tensor],  # list of [L] bool
    B: int,
    L: int,
) -> torch.Tensor:
    out = torch.zeros(B, L, 3, dtype=flat_forces.dtype, device=flat_forces.device)
    cursor = 0
    for b in range(B):
        keep = keep_list[b]
        n_keep = int(keep.sum().item())
        out[b, keep] = flat_forces[cursor:cursor + n_keep]
        cursor += n_keep
    return out


class TorchMDCGGuidance:
    """
    Wrap the paper's Cα TorchMD-Net checkpoint as a differentiable guidance force:
        F_ext = F_mut - F_wt
    where F = -dE/dx for the same coordinates x but different residue embeddings.
    """

    def __init__(
        self,
        checkpoint: str,
        mutations: Optional[List[Dict]] = None,
        xhat_is_nm: bool = True,
        use_mutant_minus_wt: bool = True,
        score_clip: float = 50.0,
        guidance_scale: float = 1.0,
    ):
        self.checkpoint = checkpoint
        self.mutations = mutations or []
        self.xhat_is_nm = xhat_is_nm
        self.use_mutant_minus_wt = use_mutant_minus_wt
        self.score_clip = float(score_clip)
        self.guidance_scale = float(guidance_scale)

        self.model = None
        self.model_device = None

    def _ensure_model(self, device: torch.device):
        if self.model is not None and self.model_device == device:
            return

        from torchmdnet.models.model import create_model
        
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        
        if "hyper_parameters" in ckpt:
            args = dict(ckpt["hyper_parameters"])
        elif "args" in ckpt:
            args = dict(ckpt["args"])
        else:
            raise KeyError(
                "Checkpoint does not contain 'hyper_parameters' or 'args'. "
                f"Top-level keys are: {list(ckpt.keys())}"
            )
            
        compat_defaults = {
            "aggr": "add",
            "derivative": True,
        }
        for k, v in compat_defaults.items():
            args.setdefault(k, v)

        # Force derivative mode because we need forces/gradients.
        args["derivative"] = True

        if not hasattr(self, "_debug_ckpt_printed"):
            print("\n[TORCHMD DEBUG] checkpoint:", self.checkpoint)
            print("[TORCHMD DEBUG] checkpoint top-level keys:", list(ckpt.keys()))
            print("[TORCHMD DEBUG] hyperparameter keys:", sorted(args.keys()))
            self._debug_ckpt_printed = True

        model = create_model(args)

        # state_dict = ckpt["state_dict"]
        # if any(k.startswith("model.") for k in state_dict.keys()):
        #     state_dict = {
        #         (k[6:] if k.startswith("model.") else k): v
        #         for k, v in state_dict.items()
        #     }
            
        state_dict = ckpt["state_dict"]

        # Strip lightning "model." prefix if present
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }

        # Compatibility remap for older checkpoints where output_network was a plain Sequential
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("output_model.output_network."):
                parts = k.split(".")
                # output_model.output_network.0.weight -> output_model.output_network.layers.0.weight
                if len(parts) >= 4 and parts[2].isdigit():
                    k = "output_model.output_network.layers." + ".".join(parts[2:])
            remapped[k] = v

        state_dict = remapped

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("[TORCHMD DEBUG] load_state_dict missing key count:", len(missing))
        if len(missing) > 0:
            print("[TORCHMD DEBUG] first missing keys:", missing[:20])

        print("[TORCHMD DEBUG] load_state_dict unexpected key count:", len(unexpected))
        if len(unexpected) > 0:
            print("[TORCHMD DEBUG] first unexpected keys:", unexpected[:20])

        self.model = model.to(device)
        self.model.eval()
        self.model_device = device


    def build_mutant_aatype(
        self,
        wt_aatype: torch.Tensor,                # [B, L]
        residue_index: Optional[torch.Tensor] = None,   # [B, L]
        chain_index: Optional[torch.Tensor] = None,     # [B, L]
    ) -> torch.Tensor:
        """
        Supported mutation specs:
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
                mask = (residue_index == int(spec["resid"]))
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
                    f"WT residue mismatch for spec {spec}. "
                    f"Observed AF2 indices {obs_unique}, expected {wt_expected}."
                )

            mut_aatype[mask] = mut_target

        return mut_aatype

    def compute_force(
        self,
        x_hat: torch.Tensor,                    # [B, L, 3], current code's scaled coordinates
        wt_aatype: torch.Tensor,               # [B, L], local AF2/OpenFold indexing
        mut_aatype: torch.Tensor,              # [B, L], local AF2/OpenFold indexing
        residue_mask: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:
        """
        Returns force in the SAME coordinate units as x_hat.
        If x_hat is numerically in nm, convert coords to Å before model call,
        then convert returned force back from per-Å to per-nm.
        """
        self._ensure_model(x_hat.device)
        model_dtype = next(self.model.parameters()).dtype

        B, L, _ = x_hat.shape

        if self.xhat_is_nm:
            pos_for_model = x_hat * 10.0   # nm -> Å
            force_backscale = 10.0         # dE/d(nm) = 10 * dE/d(Å)
        else:
            pos_for_model = x_hat
            force_backscale = 1.0
        
        pos_for_model = pos_for_model.to(dtype=model_dtype)

        z_wt_full = af2_aatype_to_torchmd_z(wt_aatype)
        z_mut_full = af2_aatype_to_torchmd_z(mut_aatype)

        z_wt, pos_flat, batch_idx, keep_list = _pack_graph_batch(
            pos=pos_for_model,
            z_full=z_wt_full,
            residue_mask=residue_mask,
        )
        z_mut, _, _, _ = _pack_graph_batch(
            pos=pos_for_model,
            z_full=z_mut_full,
            residue_mask=residue_mask,
        )
        
        ## DEBUG 
        if not hasattr(self, "_debug_inputs_printed"):
            ca_step = torch.norm(x_hat[0, 1:] - x_hat[0, :-1], dim=-1)
            print("\n[TORCHMD DEBUG] checkpoint:", self.checkpoint)
            print("[TORCHMD DEBUG] x_hat shape:", tuple(x_hat.shape))
            print("[TORCHMD DEBUG] x_hat neighbor distance mean/min/max:",
                float(ca_step.mean().detach().cpu()),
                float(ca_step.min().detach().cpu()),
                float(ca_step.max().detach().cpu()))
            print("[TORCHMD DEBUG] unique wt z:", torch.unique(z_wt).detach().cpu().tolist())
            print("[TORCHMD DEBUG] unique mut z:", torch.unique(z_mut).detach().cpu().tolist())
            self._debug_inputs_printed = True
        if not hasattr(self, "_debug_z_site_printed"):
            diff_mask = (wt_aatype != mut_aatype)
            print("[TORCHMD DEBUG] mutated site wt TorchMD z:", z_wt_full[diff_mask].detach().cpu().tolist())
            print("[TORCHMD DEBUG] mutated site mut TorchMD z:", z_mut_full[diff_mask].detach().cpu().tolist())
            self._debug_z_site_printed = True
        if not hasattr(self, "_debug_z_count_printed"):
            for val in [5, 9]:
                wt_count = int((z_wt_full == val).sum().item())
                mut_count = int((z_mut_full == val).sum().item())
                print(f"[TORCHMD DEBUG] z={val}: wt_count={wt_count}, mut_count={mut_count}")
            self._debug_z_count_printed = True



        try:
            out_wt = self.model(z=z_wt, pos=pos_flat, batch=batch_idx)
            out_mut = self.model(z=z_mut, pos=pos_flat, batch=batch_idx)
        except TypeError:
            out_wt = self.model(z_wt, pos_flat, batch=batch_idx)
            out_mut = self.model(z_mut, pos_flat, batch=batch_idx)

        if not isinstance(out_wt, (tuple, list)) or len(out_wt) < 2:
            raise RuntimeError(
                "Expected torchmd-net model(..., derivative=True) to return (energy, force). "
                "If your local torchmd-net build uses a different forward signature, "
                "change the call site in this wrapper accordingly."
            )

        _, F_wt_flat = out_wt
        _, F_mut_flat = out_mut

        if self.use_mutant_minus_wt:
            F_ext_flat = F_mut_flat - F_wt_flat
        else:
            F_ext_flat = F_mut_flat

        F_ext_flat = F_ext_flat * force_backscale
        F_ext = _unpack_forces(F_ext_flat, keep_list, B=B, L=L)
        F_ext = F_ext.to(dtype=x_hat.dtype)
        ## DEBUG
        if not hasattr(self, "_debug_force_printed"):
            print("[TORCHMD DEBUG] F_ext abs mean/max:",
                float(F_ext.abs().mean().detach().cpu()),
                float(F_ext.abs().max().detach().cpu()))
            self._debug_force_printed = True
        return F_ext