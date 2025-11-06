# SPLM-V2-GVP  [![paper](https://img.shields.io/badge/bioRxiv-Paper-<COLOR>.svg)](https://www.biorxiv.org/content/10.1101/2025.04.23.650337v1)

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>


**Structure-aware Protein Language Model V2**

SPLM-V2-GVP aligns a sequence encoder (e.g., ESM) with a **GVP** (Geometric Vector Perceptron) structural encoder to inject 3D knowledge into residue- and protein-level embeddings for downstream protein prediction tasks. Compared with S-PLM v1 (which used contact-map + Swin), V2 replaces the structure branch with **GVP over coordinate graphs**, providing geometry-aware features at the residue level.


> Prior work: [S-PLM v1](https://github.com/duolinwang/S-PLM)

   
## Installation

Please **install dependencies exactly in the order listed in [install.txt](https://github.com/Yichuan0712/SPLM-V2-GVP/blob/main/install.txt)**.


   
## Pretrained checkpoint

Download the pretrained SPLM-V2-GVP weights from **[this OneDrive link](https://mailmissouri-my.sharepoint.com/personal/wangdu_umsystem_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwangdu%5Fumsystem%5Fedu%2FDocuments%2FS%2DPLM%2Dmodel%2Fmodel%2Fcheckpoint%5F0280000%5Fgvp%2Epth&parent=%2Fpersonal%2Fwangdu%5Fumsystem%5Fedu%2FDocuments%2FS%2DPLM%2Dmodel%2Fmodel&ga=1)** and set its path in your config or pass it via `--checkpoint_path`.

   

## Quickstart  [![Colab quickstart](https://img.shields.io/badge/Colab-quickstart-e91e63)](#)

### Open [**SPLM_v2_GVP_quickstart.ipynb**](https://github.com/Yichuan0712/SPLM-V2-GVP/blob/main/SPLM_v2_GVP_quickstart.ipynb) for a minimal, runnable demo.

### 1) Preprocess PDB → HDF5 (GVP-ready)

Convert raw PDB structures into **HDF5 graphs/tensors** for the GVP encoder:

```bash
python data/preprocess_pdb.py \
  --data /path/to/raw_pdb_dir \
  --save_path /path/to/output_hdf5_dir \
  --max_workers 4
```

> Only the **processed HDF5** can be consumed by the structure (GVP) branch.


### 2) Embedding extraction

#### 2.1 Structure embeddings (residue-level)

Generate residue-level **structure embeddings** directly from the preprocessed HDF5:

```bash
python -m utils.generate_struct_embedding \
  --hdf5_path /path/to/output_hdf5_dir \
  --out_file protein_struct_embeddings.pkl \
  --config_path /path/to/configs/config_plddtallweight_noseq_rotary_foldseek.yaml \
  --checkpoint_path /path/to/checkpoint_0280000_gvp.pth \
  --result_path ./ \
  --residue_level
```

#### 2.2 Sequence embeddings (protein-level)

Standard run from FASTA to **protein-level** sequence embeddings:

```bash
python -m utils.generate_seq_embedding \
  --input_seq /path/to/protein.fasta \
  --out_file protein_embeddings.pkl \
  --config_path /path/to/configs/config_plddtallweight_noseq_rotary_foldseek.yaml \
  --checkpoint_path /path/to/checkpoint_0280000_gvp.pth \
  --result_path ./
```

#### 2.3 Truncated inference for long sequences

Limit the max sequence length (e.g., 1022) at inference time:

```bash
python -m utils.generate_seq_embedding \
  --input_seq /path/to/protein.fasta \
  --config_path /path/to/configs/config_plddtallweight_noseq_rotary_foldseek.yaml \
  --checkpoint_path /path/to/checkpoint_0280000_gvp.pth \
  --result_path ./ \
  --out_file truncate_protein_embeddings.pkl \
  --truncate_inference 1 \
  --max_length_inference 1022
```

#### 2.4 Residue-level sequence embeddings

Produce **residue-level** sequence embeddings (with truncation):

```bash
python -m utils.generate_seq_embedding \
  --input_seq /path/to/protein.fasta \
  --config_path /path/to/configs/config_plddtallweight_noseq_rotary_foldseek.yaml \
  --checkpoint_path /path/to/checkpoint_0280000_gvp.pth \
  --result_path ./ \
  --out_file truncate_protein_residue_embeddings.pkl \
  --truncate_inference 1 \
  --max_length_inference 1022 \
  --residue_level
```


### 3) CATH evaluation (structure branch)

Evaluate the structure branch on CATH using the provided utility:

```python
import os
from pathlib import Path
from utils.generate_struct_embedding import StructRepresentModel
from cath_with_struct import evaluate_with_cath_more_struct

# Paths (match your local paths)
checkpoint_path = "/path/to/checkpoint_0280000_gvp.pth"
config_path = "/path/to/configs/config_plddtallweight_noseq_rotary_foldseek.yaml"
cathpath = "/path/to/CATH_4_3_0_non-rep_gvp/"   # directory created by the preprocess step

# Build model from config + checkpoint
model, device, configs = StructRepresentModel(
    config_path=config_path,
    checkpoint_path=checkpoint_path
)

# Output directory for evaluation artifacts
out_figure_path = os.path.join(config_path.split(".yaml")[0], "CATH_test_release")
Path(out_figure_path).mkdir(parents=True, exist_ok=True)

# Run evaluation
scores_cath = evaluate_with_cath_more_struct(
    out_figure_path,
    device=device,
    batch_size=1,
    model=model,
    cathpath=cathpath,
    configs=configs
)

# scores_cath contains the metrics reported in the notebook (digits, ARI, silhouette, etc.)
print(scores_cath)
```


   

## Citation
If you use this code or the pretrained models, please cite the following paper:

### [1] S-PLM V2: residue-level model using a GVP structural encoder; efficient fine-tuning toolkit.
Zhang Y, Qin Y, Pourmirzaei M, Shao Q, Wang D, Xu D. Enhancing Structure-Aware Protein Language Models with Efficient Fine-Tuning for Various Protein Prediction Tasks. *Methods Mol Biol.* 2025;2941:31–58. doi:10.1007/978-1-0716-4623-6_2. PMID: 40601249.

### [2] S-PLM V1: protein-level contrastive learning, using Swin-transformer as protein structure encoder.
Wang D, Pourmirzaei M, Abbas UL, Zeng S, Manshour N, Esmaili F, Poudel B, Jiang Y, Shao Q, Chen J, Xu D. S-PLM: Structure-Aware Protein Language Model via Contrastive Learning Between Sequence and Structure. Adv Sci (Weinh). 2025 Feb;12(5):e2404212. doi: 10.1002/advs.202404212. Epub 2024 Dec 12. PMID: 39665266; PMCID: PMC11791933.

