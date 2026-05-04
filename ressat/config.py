import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial Gene Expression Prediction")

    # ── Dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, default="./data/", help="Directory for QC'ed data") 
    parser.add_argument("--section_num", type=int, default=2, help="Number of sections")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--result_dir", type=str, default="./results/", help="Directory for saving results") 

    # ── Hyper-parameters ────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=224, help="Resized patch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_fourier", type=int, default=256, help="Number of Fourier features for spatial encoder")
    parser.add_argument("--sigma", type=float, default=1, help="Sigma for Fourier feature projection")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--exp_name", type=str, default="ressat", help="Folder name for saving checkpoints")
    
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    
    args.torch_home = os.path.join(args.result_dir, ".cache", "torch")
    os.environ["TORCH_HOME"] = args.torch_home
    os.makedirs(args.torch_home, exist_ok=True)

    return args
