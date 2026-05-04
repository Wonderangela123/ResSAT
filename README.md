# ResSAT: Enhancing Spatial Transcriptomics Prediction from H&E-Stained Histology Images with Interactive Spot Transformer
ResSAT extracts visual features from the H&E patch through an image encoder and encodes spatial coordinates through a spatial encoder based on Fourier feature mapping 16, 17. The resulting image and spatial representations are integrated through a learned modulation mechanism, which allows spatial features to adaptively adjust image representations. A self-attention transformer mechanism then operates across spots to learn spatially informed interactions for gene expression prediction.

<img width="1095" height="595" alt="image" src="https://github.com/user-attachments/assets/0d6837ff-9c87-4c46-837f-f58b9a8308fb" />

## Installation
```bash
conda env create -f environment.yml -n ressat_env
conda activate ressat_env
pip install -e .
```
