import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────────────
# Spot-level records
# ─────────────────────────────────────────────────────────────────────────────
def build_spot_records(sections):
    records = []
    for sec_id, sec in enumerate(sections):
        assert len(sec["data"]) == len(sec["locs"]), \
            f"Section {sec_id}: len(data) != len(locs)"
        for spot_id, ((patch, expr), loc) in enumerate(zip(sec["data"], sec["locs"])):
            records.append({
                "image": patch,
                "expr": expr,
                "loc": loc,
                "section_id": sec_id,
                "spot_id": spot_id,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HEPatchesDataset(Dataset):
    def __init__(self, records, patch_size):
        self.records = records
        locs = np.array([r["loc"] for r in records], dtype=np.float32)
        self.locations = locs
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = self.transform(rec["image"])
        gene_expression = torch.tensor(rec["expr"], dtype=torch.float32)
        coord = torch.tensor(self.locations[idx], dtype=torch.float32)
        section_id = torch.tensor(rec["section_id"], dtype=torch.long)
        spot_id = torch.tensor(rec["spot_id"], dtype=torch.long)
        return image, gene_expression, coord, section_id, spot_id
