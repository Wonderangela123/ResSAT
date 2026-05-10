import os
import torch


def back_project(embedding, pca_info):
    """
    Project 50-dimensional Harmony-corrected embeddings back to the 2,000-HVG log-normalized expression space.
    
    Parameters
    ----------
    embedding : torch.Tensor
        Tensor of shape (N, 50), containing Harmony-corrected representations.
    
    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2000), containing reconstructed log-normalized gene expression profiles.
    """
    components = torch.tensor(pca_info["components"], dtype=torch.float32)  # (50, 2000)
    mean       = torch.tensor(pca_info["mean"],       dtype=torch.float32)  # (2000,)
    std        = torch.tensor(pca_info["std"],        dtype=torch.float32)  # (2000,)
    y_scaled   = embedding @ components       # (N, 2000)
    y_lognorm  = y_scaled * std + mean        # 反scale
    return y_lognorm


def safe_correlation(array1, array2, default_value=0.0):
    std1 = torch.std(array1)
    std2 = torch.std(array2)
    if std1 == 0 or std2 == 0:
        return default_value
    cov = torch.mean((array1 - torch.mean(array1)) * (array2 - torch.mean(array2)))
    return (cov / (std1 * std2)).item()


def evaluate(y_pred, y_test, gene_names=None):
    n_genes = y_pred.shape[1]
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    cor_list = [safe_correlation(y_pred[:, i], y_test[:, i]) for i in range(n_genes)]
    paired = sorted(zip(gene_names, cor_list), key=lambda x: x[1], reverse=True)

    return cor_list, paired


def save_results(y_pred, y_test, gene_names, result_dir):
    cor_list, paired = evaluate(y_pred, y_test, gene_names=gene_names)
    
    torch.save(y_pred, os.path.join(result_dir, "y_pred.pt"))
    torch.save(y_test, os.path.join(result_dir, "y_test.pt"))
    torch.save(paired, os.path.join(result_dir, "sorted_correlations.pt"))
    
    return cor_list, paired
