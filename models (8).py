import os
import math
import glob
import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import DataLoader
from tqdm import tqdm


class SpatialEncoder(nn.Module):
    def __init__(self, in_dim=2, num_fourier=256, sigma=10, dropout=0.1, out_dim=128):
        super().__init__()
        self.in_dim = in_dim
        self.num_fourier = num_fourier
        B = torch.randn(in_dim, num_fourier) * sigma
        self.register_buffer("B", B)
        self.mlp = nn.Sequential(
            nn.Linear(num_fourier * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, coords):
        x_proj = 2 * math.pi * coords @ self.B
        fourier_feat = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.mlp(fourier_feat)


class ResNetSpatial(nn.Module):
    def __init__(self, output_dim, num_fourier=256, sigma=10, dropout=0.1):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.img_proj = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout)
        )
        self.spatial_enc = SpatialEncoder(
            in_dim=2, num_fourier=num_fourier, sigma=sigma, dropout=dropout, out_dim=128
        )
        self.gamma = nn.Linear(128, 512)
        self.beta = nn.Linear(128, 512)

        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

        self.feature_dim = 512 + 128

    def forward(self, x, coords):
        img_feat = self.backbone(x).flatten(1)
        img_feat = self.img_proj(img_feat)
        spatial_feat = self.spatial_enc(coords)
        feat = torch.cat(
            [img_feat * self.gamma(spatial_feat) + self.beta(spatial_feat),
             spatial_feat],
            dim=1,
        )
        return feat

    
class SA_Feature(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key   = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, feat):
        """
        feat: (B, D)
        """
        B, D = feat.shape
        
        # 全局SA: batch内所有spot互相做self attention       
        seq_norm = self.norm1(feat)
        Q = self.query(seq_norm)
        K = self.key(seq_norm)
        V = self.value(seq_norm)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x_attn = torch.matmul(attn, V)
        feat = feat + self.proj(x_attn)
        out = feat + self.ffn(self.norm2(feat))
        
        return out


class PredictionHead(nn.Module):
    def __init__(self, in_dim, output_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(in_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)

    
class ResSAT:
    def __init__(
        self,
        train_sections=None,
        val_sections=None,
        test_sections=None,
        data_dir=".",
        result_dir="./results",
        patch_size=224,
        num_fourier=256,
        sigma=10,
        dropout=0.1,
        num_workers=32,
        exp_name="ressat",
        gene_names=None,
    ):
        # ── TORCH_HOME ──────────────────────────────────────
        torch_home = os.environ.get("TORCH_HOME", os.path.join(result_dir, ".cache", "torch"))
        os.environ["TORCH_HOME"] = torch_home
        os.makedirs(torch_home, exist_ok=True)

        # ── Lazy imports ────────────────────────────────────
        from ressat.dataset import build_spot_records, HEPatchesDataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Config ──────────────────────────────────────────
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.exp_name = exp_name
        self.dropout = dropout 
        self.gene_names = gene_names

        # ── Build records + datasets ──────────────────
        # Figure out output_dim from whichever sections are provided
        any_sections = train_sections or val_sections or test_sections
        first_section = any_sections[0]
        self.output_dim = len(first_section["data"][0][1])

        if train_sections is not None:
            self.train_records = build_spot_records(train_sections)
            self.train_dataset = HEPatchesDataset(self.train_records, patch_size)

        if val_sections is not None:
            self.val_records = build_spot_records(val_sections)
            self.val_dataset = HEPatchesDataset(self.val_records, patch_size)

        if test_sections is not None:
            self.test_records = build_spot_records(test_sections)
            self.test_dataset = HEPatchesDataset(self.test_records, patch_size)

        # ── Model ───────────────────────────────────────────
        self.model = ResNetSpatial(self.output_dim, num_fourier, sigma).to(self.device)
        self.sa_feature = SA_Feature(dim=self.model.feature_dim).to(self.device)
        self.pred_head = PredictionHead(self.model.feature_dim, self.output_dim, self.dropout).to(self.device)

        # ── Save dir ────────────────────────────────────────
        self.save_dir = os.path.join(result_dir, exp_name)

    # =================================================================
    # FIT
    # =================================================================
    def fit(
        self,
        num_epochs=100,
        lr=1e-4,
        batch_size=32,
        patience=10,
        weight_decay=1e-5,
    ):

        # ── Version dir ─────────────────────────────────────
        version = 0
        vdir = os.path.join(self.save_dir, "lightning_logs", f"version_{version}")
        while os.path.exists(vdir):
            version += 1
            vdir = os.path.join(self.save_dir, "lightning_logs", f"version_{version}")
        self.ckpt_dir = os.path.join(vdir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_path = os.path.join(vdir, "metrics.csv")

        device = self.device
        loader_kw = dict(num_workers=self.num_workers, pin_memory=True)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)

        if hasattr(self, "val_dataset"):
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kw)
        else:
            val_loader = None


        # ── Optimizer ───────────────────────────────────────
        params = [
            {"params": self.model.backbone[0].parameters(), "lr": lr / 10},
            {"params": self.model.backbone[1].parameters(), "lr": lr / 8},
            {"params": self.model.backbone[4].parameters(), "lr": lr / 2},
            {"params": self.model.backbone[5].parameters(), "lr": lr / 2},
            {"params": self.model.backbone[6].parameters(), "lr": lr / 2},
            {"params": self.model.backbone[7].parameters(), "lr": lr / 2},
            {"params": self.model.img_proj.parameters(), "lr": lr},
            {"params": self.model.spatial_enc.parameters(), "lr": lr},
            {"params": self.model.gamma.parameters(), "lr": lr},
            {"params": self.model.beta.parameters(), "lr": lr},
            {"params": self.sa_feature.parameters(), "lr": lr},
            {"params": self.pred_head.parameters(), "lr": lr},
        ]
        optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        best_loss = float("inf")
        epochs_no_improve = 0

        epoch_bar = tqdm(range(num_epochs))
        for epoch in epoch_bar:
            # train
            self.model.train()
            self.sa_feature.train()
            self.pred_head.train()
            train_loss = 0.0

            for x, y, coords, sec_ids, spot_ids in train_loader:
                x, y, coords = x.to(device), y.to(device), coords.to(device)
                sec_ids, spot_ids = sec_ids.to(device), spot_ids.to(device)
                optimizer.zero_grad()
                feat = self.model(x, coords)
                z = self.sa_feature(feat)
                y_pred = self.pred_head(z)
                loss = torch.mean((y - y_pred) ** 2)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # validation
            if val_loader is not None:
                self.model.eval()
                self.sa_feature.eval()
                self.pred_head.eval()

                val_loss = 0
                with torch.no_grad():
                    for x, y, coords, sec_ids, spot_ids in val_loader:
                        x, y, coords = x.to(device), y.to(device), coords.to(device)
                        sec_ids, spot_ids = sec_ids.to(device), spot_ids.to(device)
                        feat = self.model(x, coords)
                        z = self.sa_feature(feat)
                        y_pred = self.pred_head(z)
                        val_loss += torch.mean((y - y_pred) ** 2).item()
                val_loss /= len(val_loader)

                epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

                with open(self.log_path, "a") as f:
                    f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f}\n")

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    for old in glob.glob(os.path.join(self.ckpt_dir, "best-*.ckpt")):
                        os.remove(old) 
                    name = f"best-epoch={epoch:02d}-val_loss={best_loss:.4f}.ckpt"
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "sa_feature_state": self.sa_feature.state_dict(),
                        "pred_head_state": self.pred_head.state_dict(),
                        "val_loss": val_loss,
                    }, os.path.join(self.ckpt_dir, name))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        epoch_bar.close()
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                        
            else:
                val_loss = None

                epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")

                with open(self.log_path, "a") as f:
                    f.write(f"{epoch+1},{train_loss:.4f}\n")

                if train_loss < best_loss:
                    best_loss = train_loss
                    epochs_no_improve = 0

                    for old in glob.glob(os.path.join(self.ckpt_dir, "best-*.ckpt")):
                        os.remove(old)

                    name = f"best-epoch={epoch:02d}-train_loss={best_loss:.4f}.ckpt"

                    torch.save({
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "sa_feature_state": self.sa_feature.state_dict(),
                        "pred_head_state": self.pred_head.state_dict(),
                        "train_loss": train_loss,
                    }, os.path.join(self.ckpt_dir, name))

                else:
                    epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        epoch_bar.close()
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            scheduler.step()

        print(f"Checkpoints in: {self.ckpt_dir}")

    # =================================================================
    # LOAD CHECKPOINT
    # =================================================================
    def load_checkpoint(self, ckpt_path=None):
        """Load a checkpoint. If None, finds the latest best checkpoint."""
        if ckpt_path is None:
            pattern = os.path.join(self.save_dir, "lightning_logs", "version_*", "checkpoints", "best-*.ckpt")
            ckpts = glob.glob(pattern)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {self.save_dir}")
            ckpt_path = sorted(ckpts, key=os.path.getmtime)[-1]

        print(f"Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.sa_feature.load_state_dict(ckpt["sa_feature_state"])
        self.pred_head.load_state_dict(ckpt["pred_head_state"])

    # =================================================================
    # PREDICT
    # =================================================================
    def predict(self, batch_size=32):
        """Run on test set. Call load_checkpoint() first.
        Returns (y_pred, y_true) as numpy arrays."""
        if not hasattr(self, "test_dataset"):
            raise RuntimeError("No test data. Pass test_sections to ResSAT().")

        device = self.device
        loader_kw = dict(num_workers=self.num_workers, pin_memory=True)

        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kw)

        self.model.eval()
        self.sa_feature.eval()
        self.pred_head.eval()

        # predict
        test_preds, test_trues = [], []
        with torch.no_grad():
            for x, y, coords, sec_ids, spot_ids in tqdm(test_loader, desc="Predicting", leave=False):
                x, y, coords = x.to(device), y.to(device), coords.to(device)
                sec_ids, spot_ids = sec_ids.to(device), spot_ids.to(device)
                feat = self.model(x, coords)
                z = self.sa_feature(feat)
                y_pred = self.pred_head(z)
                test_preds.append(y_pred.cpu())
                test_trues.append(y.cpu())

        self.y_pred_test = torch.cat(test_preds)
        self.y_test = torch.cat(test_trues)

        return self.y_pred_test, self.y_test
