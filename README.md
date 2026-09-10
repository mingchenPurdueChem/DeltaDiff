# DeltaDiff

DeltaDiff is a physics-guided inference workflow for mutation-aware protein conformational sampling with a pretrained structure-to-structure diffusion model and coarse-grained guidance from either TorchMD-Net or MLCG.

The workflow uses:
- a pretrained **Str2Str** checkpoint as the base diffusion model
- an external mutation-guidance potential from either a pretrained **TorchMD coarse-grained multiprotein checkpoint** or a pretrained **MLCG (ClementiGroup) transferable coarse-grained checkpoint**

During reverse diffusion, DeltaDiff evaluates the same evolving structure with both wild-type and mutant sequence embeddings in the chosen CG model, computes the mutant-minus-WT force difference, and injects that guidance back into the reverse update. The two backends are drop-in alternatives for each other: same interface, same `F_ext = F_mut - F_wt` guidance contract, same pull-back-to-noise-space usage in `src/models/score/r3.py` — only the force-field evaluator differs.

---

## Environment

Create the main environment with:

    conda env create -f env.yml
    conda activate deltadiff

---

## Checkpoints

Large model weights are not included in this repository copy.

### 1) Str2Str pretrained diffusion checkpoint

DeltaDiff uses the pretrained Str2Str diffusion checkpoint as the base structure-to-structure model.

The original Str2Str pretrained checkpoint is distributed through the authors' Google Drive link:

    https://drive.google.com/file/d/1YsvFXOpdst4QxK34GSWvLjgbvzUq4Ry8/view

Download the checkpoint and place it at:

    src/ckpt/pretrain.pth

or, equivalently through the Hydra path setting:

    ${paths.data_dir}/ckpt/pretrain.pth

If preferred, the file can also be downloaded from the command line with `gdown`:

    pip install gdown
    gdown 1YsvFXOpdst4QxK34GSWvLjgbvzUq4Ry8 -O src/ckpt/pretrain.pth

Make sure the checkpoint filename matches the path expected by the current configuration before running evaluation or inference workflows.

### 2) TorchMD coarse-grained guidance checkpoint

The mutation-guidance module uses the pretrained coarse-grained protein thermodynamics model from the official `torchmd-protein-thermodynamics` repository.

Download the TorchMD model archive with:

    git clone https://github.com/torchmd/torchmd-protein-thermodynamics
    cd torchmd-protein-thermodynamics/Models
    wget pub.htmd.org/protein_thermodynamics_data/Models.zip
    unzip Models.zip

After extracting the archive, DeltaDiff uses the multiprotein checkpoint:

    Models/multiprotein/model.ckpt

Update the path in `configs/model/diffusion.yaml` to point to your local copy of this checkpoint before running mutation-guided workflows.

### 3) MLCG coarse-grained guidance checkpoint (alternative to TorchMD)

As an alternative external guidance potential, DeltaDiff can instead use the pretrained transferable coarse-grained model from Charron, N. E. et al., ["Navigating protein landscapes with a machine-learned transferable coarse-grained model"](https://www.nature.com/articles/s41557-025-01874-0) (*Nat. Chem.*, 2025), built on the [ClementiGroup/mlcg](https://github.com/ClementiGroup/mlcg) package.

First, install the `mlcg` package into the `deltadiff` environment:

    git clone https://github.com/ClementiGroup/mlcg
    cd mlcg
    pip install .

Then download the pretrained model-and-prior checkpoint from the paper's Zenodo record:

    https://doi.org/10.5281/zenodo.15465782

and place the `model_and_prior.pt` file at:

    mlcg_pretrained/checkpoint/model_and_prior.pt

Point the guidance module at this checkpoint the same way as the TorchMD one (see `load_mlcg_model` / `MLCGGuidance` in `scripts/mlcg_guidance.py`, and `src/models/guidance/mlcg_cg_guidance.py` for the drop-in `TorchMDCGGuidance`-compatible wrapper used in `configs/model/diffusion.yaml`).

If you need to guide sampling for a protein sequence not already covered by the provided checkpoint inputs, use [mlcg-tk](https://github.com/ClementiGroup/mlcg-tk) to generate the required per-molecule configuration/template files — follow the instructions in the `examples` folder of that repo (the transferable model's prior parameters are at `mlcg-tk/examples/transferable_priors.yaml`).

---

## Input Data

Evaluation and inference read input structures from the test-data path configured through Hydra:

    ${paths.test_data_path}

which is defined from the environment variable:

    TEST_DATA

A simple workflow is to place the input PDB(s) in a folder and export that folder as `TEST_DATA`:

    export TEST_DATA=/path/to/test_pdb_folder

If you want to test a single structure, the easiest approach is to create a folder containing only that one PDB.

---
