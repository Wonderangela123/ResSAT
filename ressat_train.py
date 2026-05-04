#!/usr/bin/env python
from ressat.config import parse_args
from ressat.data_loader import load_sections, load_gene_names
from ressat.models import ResSAT
import torch
import numpy as np
import random
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)
    
def main():
    args = parse_args()

    train_sections, val_sections, _ = load_sections(data_dir=args.data_dir, section_num=args.section_num)
    gene_names = load_gene_names(data_dir=args.data_dir)

    model = ResSAT(
        train_sections=train_sections,
        val_sections=val_sections,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        gene_names=gene_names,
        patch_size=args.patch_size,
        num_fourier=args.num_fourier,
        sigma=args.sigma,
        dropout=args.dropout,
        num_workers=args.num_workers,
        exp_name=args.exp_name,
    )

    model.fit(
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )


# # only for mouse brain dataset
# def main():
#     args = parse_args()
    
#     seeds = [0, 1, 42, 123, 12345]
    
#     for i, seed in enumerate(seeds, start=1):
#         print(f"\n{'='*60}")
#         print(f"Running rep{i} with seed: {seed}")
#         print(f"{'='*60}\n")
        
#         set_seed(seed)
        
#         train_sections, val_sections, _ = load_sections(
#             data_dir=args.data_dir, 
#             section_num=args.section_num
#         )
#         gene_names = load_gene_names(data_dir=args.data_dir)
        
#         exp_name_with_rep = os.path.join(args.exp_name, f"rep{i}")
        
#         model = ResSAT(
#             train_sections=train_sections,
#             val_sections=val_sections,
#             data_dir=args.data_dir,
#             result_dir=args.result_dir,
#             gene_names=gene_names,
#             patch_size=args.patch_size,
#             num_fourier=args.num_fourier,
#             sigma=args.sigma,
#             dropout=args.dropout,
#             num_workers=args.num_workers,
#             exp_name=exp_name_with_rep,
#         )
        
#         model.fit(
#             num_epochs=args.num_epochs,
#             lr=args.lr,
#             batch_size=args.batch_size,
#             patience=args.patience,
#             weight_decay=args.weight_decay,
#         )

if __name__ == "__main__":
    main()
