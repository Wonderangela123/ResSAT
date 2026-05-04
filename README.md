# ResSAT: Enhancing Spatial Transcriptomics Prediction from H&E-Stained Histology Images with Interactive Spot Transformer
ResSAT extracts visual features from the H&E patch through an image encoder and encodes spatial coordinates through a spatial encoder based on Fourier feature mapping 16, 17. The resulting image and spatial representations are integrated through a learned modulation mechanism, which allows spatial features to adaptively adjust image representations. A self-attention transformer mechanism then operates across spots to learn spatially informed interactions for gene expression prediction.

<img width="1095" height="595" alt="image" src="https://github.com/user-attachments/assets/0d6837ff-9c87-4c46-837f-f58b9a8308fb" />

## Installation
```bash
cd ResSAT/
conda env create -f environment.yml -n ressat_env
conda activate ressat_env
pip install -e .
```

## Downloading Dataset
By default, the processed files can be downloaded from [here](https://drive.google.com/drive/folders/xxxxx) and will then be saved to `./data/`. Detailed preprocessing procedures are described in our paper.

## Training
```bash
python ressat_train.py \
    --data_dir ./data/ \
    --result_dir ./results/ \
    --section_num 2 \
    --num_fourier 128 \
    --sigma 1 \
    --batch_size 32 \
    --patch_size 224 \
    --lr 5e-4 \
    --dropout 0.3 \
    --weight_decay 1e-6 \
    --num_epochs 10 \
    --patience 1 \
    --num_workers 32
```
## Predicting
```bash
python ressat_predict.py \
    --data_dir ./data/ \
    --result_dir ./results/ \
    --section_num 2 \
    --num_fourier 128 \
    --sigma 1 \
    --batch_size 32 \
    --patch_size 224 \
    --dropout 0.3 \
    --num_workers 32
```
