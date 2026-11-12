"""Per-sample predictions for every split. These files are the raw material for AES, ensembles and stats.

Columns: source, split, row, stay_id, time, y, logit. `time` is the prediction hour (static tasks: last hour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data.dataset import IcuArrayDataset
from ..utils.io import atomic_write_parquet


@torch.no_grad()
def predict_deep(net: torch.nn.Module, dirs, batch_size: int = 512, use_mask: bool = True,
                 device: str | None = None) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    frames = []
    for source, split, path in dirs:
        ds = IcuArrayDataset(path, use_mask=use_mask)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = net(batch).float()
            rows = batch["row"].cpu().numpy()
            if logits.dim() == 2:
                mask = (batch["ymask"] & batch["valid"]).cpu().numpy()
                r, t = np.nonzero(mask)
                frames.append(pd.DataFrame({
                    "source": source, "split": split, "row": rows[r], "time": t,
                    "y": batch["y"].cpu().numpy()[r, t].astype(int), "logit": logits.cpu().numpy()[r, t]}))
            else:
                last = batch["valid"].sum(1).cpu().numpy() - 1
                frames.append(pd.DataFrame({
                    "source": source, "split": split, "row": rows, "time": last,
                    "y": batch["y"].cpu().numpy().astype(int), "logit": logits.cpu().numpy()}))
    out = pd.concat(frames, ignore_index=True)
    return attach_ids(out, dirs)


def attach_ids(pred: pd.DataFrame, dirs) -> pd.DataFrame:
    parts = []
    for source, split, path in dirs:
        idx = pd.read_parquet(path / "index.parquet")[["stay_id", "hospital_id"]].reset_index(drop=True)
        sub = pred[(pred.source == source) & (pred.split == split)]
        rows = sub["row"].to_numpy().astype(np.int64)
        sub = sub.assign(stay_id=idx["stay_id"].to_numpy()[rows],
                         hospital_id=idx["hospital_id"].to_numpy()[rows])
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def save_predictions(df: pd.DataFrame, path) -> None:
    df = df.copy()
    df["row"] = df["row"].astype(np.int64)      # position in the split's arrays
    df["time"] = df["time"].astype(np.int64)    # hour of the prediction (last valid hour for static tasks)
    df["p"] = 1 / (1 + np.exp(-df["logit"].astype(float)))
    atomic_write_parquet(df, path)
