# DeltaDiff

DeltaDiff is a physics-guided inference workflow for mutation-aware protein conformational sampling with a pretrained structure-to-structure diffusion model and TorchMD-Net coarse-grained guidance.

The workflow uses:
- a pretrained **Str2Str** checkpoint as the base diffusion model
- a pretrained **TorchMD coarse-grained multiprotein checkpoint** as an external mutation-guidance potential

During reverse diffusion, DeltaDiff evaluates the same evolving structure with both wild-type and mutant sequence embeddings in the TorchMD CG model, computes the mutant-minus-WT force difference, and injects that guidance back into the reverse update.

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
