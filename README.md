# ResSAT: Enhancing Spatial Transcriptomics Prediction from H&E-Stained Histology Images with Interactive Spot Transformer
ResSAT extracts visual features from the H&E patch through an image encoder and encodes spatial coordinates through a spatial encoder based on Fourier feature mapping 16, 17. The resulting image and spatial representations are integrated through a learned modulation mechanism, which allows spatial features to adaptively adjust image representations. A self-attention transformer mechanism then operates across spots to learn spatially informed interactions for gene expression prediction.

<img width="1095" height="595" alt="image" src="https://github.com/user-attachments/assets/0d6837ff-9c87-4c46-837f-f58b9a8308fb" />

## Installation
```bash
git clone https://github.com/Wonderangela123/ResSAT.git

cd ResSAT/

conda env create -f environment.yml -n ressat_env
conda activate ressat_env
pip install -e .
```

## Downloading Dataset
The processed files can be downloaded from [here](https://doi.org/10.5281/zenodo.20031209) and will then be saved to `./data/`. Detailed preprocessing procedures are described in our paper.

## Example Data Structure

* `data/`
   * `Section_1/`
      * `dataset.pkl` : list of `(patch, harmony_embedding)` tuples per spot, where `patch` is the H&E image patch and `harmony_embedding` is the 50-dim batch-corrected PCA embedding for gene expression.
      * `locations.pkl` : list of normalized `(x, y)` coordinates in `[0, 1]` for each spot, in the same order as `dataset.pkl`.
   * `Section_2/`
      * (same structure as `Section_1/`)
   * `pca_info.pkl` : dictionary containing PCA parameters
      * `mean` : per-gene mean used in scaling, shape `(2000,)`
      * `std` : per-gene std used in scaling, shape `(2000,)`
      * `components` : PCA loadings, shape `(50, 2000)`
      * `gene_names` : list of 2,000 highly variable gene names (same order as columns in `components`)
   * `gene_list.csv` : list of the 2,000 highly variable genes selected globally across all sections

`*` := with CLI (command line interface). Do `python ressat_train.py -h` to see all available options.

## Training
```bash
cd ressat/

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
    --num_epochs 100 \
    --patience 10 \
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
