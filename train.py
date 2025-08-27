# train.py (version finale avec embeddings élargis + optimisations)
import argparse
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

# -----------------------------
# Dataset PyTorch
# -----------------------------
class RatingsDataset(Dataset):
    def __init__(self, user_idx, item_idx, ratings):
        self.user_idx = torch.tensor(user_idx, dtype=torch.long)
        self.item_idx = torch.tensor(item_idx, dtype=torch.long)
        self.ratings  = torch.tensor(ratings,  dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, i):
        return self.user_idx[i], self.item_idx[i], self.ratings[i]


# -----------------------------
# Modèle MF (embeddings + biais)
# -----------------------------
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=128):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias    = nn.Embedding(n_users, 1)
        self.item_bias    = nn.Embedding(n_items, 1)

        # init
        nn.init.normal_(self.user_factors.weight, 0, 0.05)
        nn.init.normal_(self.item_factors.weight, 0, 0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u = self.user_factors(users)
        v = self.item_factors(items)
        dot = (u * v).sum(dim=1, keepdim=True)
        out = dot + self.user_bias(users) + self.item_bias(items)
        return out.squeeze(1)  # raw score


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


# -----------------------------
# Chargement + mapping
# -----------------------------
def load_and_prepare_ratings(path_ratings, sample_frac=1.0, seed=42):
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    usecols = ["userId", "movieId", "rating", "timestamp"]
    ratings = pd.read_csv(path_ratings, usecols=usecols, dtype=dtypes)

    if sample_frac < 1.0:
        ratings = ratings.sample(frac=sample_frac, random_state=seed)

    user_codes, user_uniques = pd.factorize(ratings["userId"], sort=True)
    item_codes, item_uniques = pd.factorize(ratings["movieId"], sort=True)

    ratings = ratings.assign(
        user_idx=user_codes.astype("int32"),
        item_idx=item_codes.astype("int32")
    ).sort_values("timestamp", ignore_index=True)

    return ratings, user_uniques.to_numpy(), item_uniques.to_numpy()


def build_dataloaders(ratings_df, val_ratio=0.1, batch_size=16384, seed=42):
    ds = RatingsDataset(
        user_idx=ratings_df["user_idx"].values,
        item_idx=ratings_df["item_idx"].values,
        ratings =ratings_df["rating"].values
    )
    n_val = int(len(ds) * val_ratio)
    n_tr  = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr_ds, val_ds = random_split(ds, [n_tr, n_val], generator=gen)

    dl_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    try:
        dl_kwargs["persistent_workers"] = True
    except Exception:
        pass

    tr_loader  = DataLoader(tr_ds, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return tr_loader, val_loader


# -----------------------------
# Normalisation des notes
# -----------------------------
def rating_to_norm(x):
    return (x - 0.5) / 4.5  # map 0.5..5.0 -> 0..1

def norm_to_rating(x):
    return x * 4.5 + 0.5   # map 0..1 -> 0.5..5.0


# -----------------------------
# Entraînement
# -----------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device utilisé : {device}")

    ratings_df, user_ids, item_ids = load_and_prepare_ratings(
        args.ratings, sample_frac=args.sample_frac, seed=args.seed
    )
    n_users = int(ratings_df["user_idx"].max() + 1)
    n_items = int(ratings_df["item_idx"].max() + 1)
    print(f"[INFO] n_users={n_users:,} | n_items={n_items:,} | n_ratings={len(ratings_df):,}")

    tr_loader, val_loader = build_dataloaders(
        ratings_df,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed
    )

    model = MatrixFactorization(n_users, n_items, n_factors=args.factors).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    artifacts = Path(args.out_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    sample_check = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        first_batch = True
        for users, items, ratings in pbar:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)

            targets = rating_to_norm(ratings)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                raw = model(users, items)
                preds_norm = torch.sigmoid(raw)
                loss = loss_fn(preds_norm, targets)

            if first_batch and epoch == 1:
                with torch.no_grad():
                    print("[DEBUG] Targets first 8:", targets[:8].cpu().numpy())
                    print("[DEBUG] Preds first 8:", preds_norm[:8].detach().cpu().numpy())
                sample_check = (users[:8].cpu().numpy(), items[:8].cpu().numpy())
                first_batch = False

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * users.size(0)
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_rmses = []
            for users, items, ratings in val_loader:
                users = users.to(device, non_blocking=True)
                items = items.to(device, non_blocking=True)
                ratings = ratings.to(device, non_blocking=True)
                targets = rating_to_norm(ratings)
                raw = model(users, items)
                preds_norm = torch.sigmoid(raw)
                val_losses.append(loss_fn(preds_norm, targets).item() * users.size(0))
                preds_rating = norm_to_rating(preds_norm)
                val_rmses.append(rmse(preds_rating, ratings).item() * users.size(0))

            val_loss = np.sum(val_losses) / len(val_loader.dataset)
            val_rmse = np.sum(val_rmses) / len(val_loader.dataset)

        print(f"[VAL] epoch={epoch}  loss={val_loss:.6f}  RMSE={val_rmse:.4f}")

        if epoch == 1 and sample_check is not None:
            us, its = sample_check
            us_t = torch.tensor(us, dtype=torch.long, device=device)
            its_t = torch.tensor(its, dtype=torch.long, device=device)
            with torch.no_grad():
                raw = model(us_t, its_t)
                preds_norm = torch.sigmoid(raw)
                print("[DEBUG] Preds after epoch1:", preds_norm[:8].cpu().numpy())
                print("[DEBUG] Remapped:", norm_to_rating(preds_norm[:8].cpu().numpy()))

        # Sauvegarde du meilleur modèle
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            ckpt = {
                "state_dict": model.state_dict(),
                "n_users": n_users,
                "n_items": n_items,
                "factors": args.factors
            }
            torch.save(ckpt, artifacts / "model.pt")
            with open(artifacts / "mappings.pkl", "wb") as f:
                pickle.dump({"user_ids": user_ids, "item_ids": item_ids}, f)
            with open(artifacts / "config.json", "w") as f:
                json.dump(vars(args), f, indent=2)
            print(f"[SAVE] Nouveau meilleur modèle → {artifacts/'model.pt'} (RMSE={best_rmse:.4f})")

    print("[DONE] Entraînement terminé.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=str, default="./ml-32m/ratings.csv")
    ap.add_argument("--out_dir", type=str, default="./artifacts")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16384)
    ap.add_argument("--factors", type=int, default=256, help="Dim embedding (128, 256, 512)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    args = ap.parse_args()
    train(args)
