#!/usr/bin/env python
from ressat.config import parse_args
from ressat.data_loader import load_sections, load_gene_names
from ressat.models import ResSAT
from ressat.utils import back_project, safe_correlation, evaluate, save_results

import numpy as np
import torch
from scipy import stats
import os
import pickle


def main():
    args = parse_args()

    _, _, test_sections = load_sections(
        data_dir=args.data_dir, section_num=args.section_num
    )
    gene_names = load_gene_names(data_dir=args.data_dir)

    model = ResSAT(
        test_sections=test_sections,
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

    model.load_checkpoint()
    y_pred_pca, y_test_pca = model.predict(batch_size=args.batch_size)
    
    
    # ── Batch effect 反推 ──────────────────────────────────
    pca_info_path = os.path.join(args.data_dir, "pca_info.pkl")
    with open(pca_info_path, "rb") as f:
        pca_info = pickle.load(f)
    y_pred = back_project(y_pred_pca, pca_info)   # (N, 2000)
    y_test = back_project(y_test_pca, pca_info)   # (N, 2000)
    print("Harmony back-projection done → gene space")
   
    cor_list, paired = save_results(y_pred, y_test, gene_names, model.save_dir) 
    
    mean_cor = np.mean(cor_list)

    mean_expr = y_test.mean(dim=0)
    top50_idx = torch.argsort(mean_expr, descending=True)[:50]
    cor_top50 = [safe_correlation(y_pred[:, i], y_test[:, i]) for i in top50_idx.tolist()]
    mean_cor_top50 = np.mean(cor_top50)

    top5_str = ", ".join([f"{g}: {v:.4f}" for g, v in paired[:5]])

    print(f"Mean Pearson R ({len(gene_names)} genes) : {mean_cor:.4f}")
    print(f"Mean Pearson R (Top-50 HEG) : {mean_cor_top50:.4f}")
    print(f"Top-5 genes : {top5_str}")

    
# def main():
#     args = parse_args()

#     _, _, test_sections = load_sections(
#         data_dir=args.data_dir, section_num=args.section_num
#     )
#     gene_names = load_gene_names(data_dir=args.data_dir)

#     # Load pca_info once
#     pca_info_path = os.path.join(args.data_dir, "pca_info.pkl")
#     with open(pca_info_path, "rb") as f:
#         pca_info = pickle.load(f)

#     # Loop through 5 reps
#     for i in range(1, 6):
#         print(f"\n{'='*60}")
#         print(f"Predicting rep{i}")
#         print(f"{'='*60}\n")
        
#         exp_name_with_rep = os.path.join(args.exp_name, f"rep{i}")
        
#         model = ResSAT(
#             test_sections=test_sections,
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

#         model.load_checkpoint()
#         y_pred_pca, y_test_pca = model.predict(batch_size=args.batch_size)
        
#         torch.save(y_pred_pca, os.path.join(model.save_dir, "y_pred_pca.pt"))
#         torch.save(y_test_pca, os.path.join(model.save_dir, "y_test_pca.pt"))

#         # Back project to gene space
#         y_pred = back_project(y_pred_pca, pca_info)
#         y_test = back_project(y_test_pca, pca_info)

#         # Save results
#         cor_list, paired = save_results(y_pred, y_test, gene_names, model.save_dir) 
#         mean_cor = np.mean(cor_list)

#         mean_expr = y_pred.mean(dim=0)
#         top50_idx = torch.argsort(mean_expr, descending=True)[:50]
#         cor_top50 = [safe_correlation(y_pred[:, i], y_test[:, i]) for i in top50_idx.tolist()]
#         mean_cor_top50 = np.mean(cor_top50)
        
#         top5_str = ", ".join([f"{g}: {v:.4f}" for g, v in paired[:5]])

#         print(f"Rep{i} - Mean Pearson R: {mean_cor:.4f}")
#         print(f"Rep{i} - Mean Pearson R (Top-50 HEG): {mean_cor_top50:.4f}")
#         print(f"Top-5 genes : {top5_str}")


if __name__ == "__main__":
    main()
