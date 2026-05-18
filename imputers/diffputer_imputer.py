from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np


def _ensure_diffputer_on_path(diffputer_root: str | None = None) -> Path:
    repo_root = (
        Path(diffputer_root)
        if diffputer_root is not None
        else Path(__file__).resolve().parents[1] / "DiffPuter"
    ).resolve()

    if not (repo_root / "main.py").is_file() or not (repo_root / "model.py").is_file():
        raise FileNotFoundError(
            f"DiffPuter repo not found at {repo_root}. "
            "Pass diffputer_root='/path/to/DiffPuter' explicitly."
        )

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


class DiffPuterImputer:
    name = "DiffPuter"

    def __init__(
        self,
        diffputer_root: str | None = None,
        n_em_iterations: int = 5,
        n_train_epochs: int = 10000,
        n_sampling_trials: int = 10,
        n_diffusion_steps: int = 50,
        hid_dim: int = 1024,
        batch_size: int = 4096,
        early_stopping_patience: int = 500,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.diffputer_root = diffputer_root
        self.n_em_iterations = n_em_iterations
        self.n_train_epochs = n_train_epochs
        self.n_sampling_trials = n_sampling_trials
        self.n_diffusion_steps = n_diffusion_steps
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self._device_arg = device
        self.verbose = verbose

    def _resolve_device(self):
        import torch
        if self._device_arg is not None:
            return self._device_arg
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit_transform(self, X: np.ndarray, seed: int) -> np.ndarray:
        _ensure_diffputer_on_path(self.diffputer_root)

        import torch
        from torch.utils.data import DataLoader
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        from model import MLPDiffusion, Model  # type: ignore[import-not-found]
        from diffusion_utils import impute_mask  # type: ignore[import-not-found]

        device = self._resolve_device()
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_samples, n_features = X.shape
        observed_mask_np = ~np.isnan(X)
        missing_mask_np = np.isnan(X)

        X_scaled = X / 2.0

        X_init = X_scaled.copy()
        X_init[missing_mask_np] = 0.0
        X_torch = torch.tensor(X_init, dtype=torch.float32)

        mask_torch = torch.tensor(missing_mask_np.astype(np.float32))

        current_filled = X_torch.clone()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for em_iter in range(self.n_em_iterations):
                # M-step: train diffusion on currently-imputed data
                # Iter 0: feed observed-only data (zeros at missing).
                # Iter 1+: feed previous iteration's full reconstruction.
                if em_iter == 0:
                    train_input = (1.0 - mask_torch) * current_filled
                else:
                    train_input = current_filled

                train_loader = DataLoader(
                    train_input.numpy(),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                )

                denoise_fn = MLPDiffusion(n_features, self.hid_dim).to(device)
                model = Model(denoise_fn=denoise_fn, hid_dim=n_features).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=1e-4, weight_decay=0,
                )
                scheduler = ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.9, patience=50,
                )

                model.train()
                best_loss = float("inf")
                patience = 0
                best_state = None
                final_epoch = 0

                for epoch in range(self.n_train_epochs):
                    final_epoch = epoch
                    batch_loss = 0.0
                    n_seen = 0
                    for batch in train_loader:
                        inputs = batch.float().to(device)
                        loss = model(inputs).mean()
                        batch_loss += loss.item() * len(inputs)
                        n_seen += len(inputs)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    curr_loss = batch_loss / max(n_seen, 1)
                    scheduler.step(curr_loss)

                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        patience = 0
                        best_state = {
                            k: v.detach().clone()
                            for k, v in model.state_dict().items()
                        }
                    else:
                        patience += 1
                        if patience >= self.early_stopping_patience:
                            break

                if best_state is not None:
                    model.load_state_dict(best_state)
                model.eval()

                if self.verbose:
                    print(
                        f"  [DiffPuter EM {em_iter + 1}/{self.n_em_iterations}] "
                        f"trained {final_epoch + 1} epochs, best loss {best_loss:.4f}"
                    )

                # E-step: conditional sampling for missing entries.
                # Pass observed-only data; impute_mask handles the rest.
                impute_X = ((1.0 - mask_torch) * current_filled).to(device)
                mask_dev = mask_torch.to(device)
                net = model.denoise_fn_D

                rec_Xs = []
                for _trial in range(self.n_sampling_trials):
                    rec_X = impute_mask(
                        net,
                        impute_X,
                        mask_torch,
                        n_samples,
                        n_features,
                        num_steps=self.n_diffusion_steps,
                        device=device,
                    )
                    # Observed (mask=0): keep impute_X. Missing (mask=1): rec_X.
                    combined = rec_X * mask_dev + impute_X * (1.0 - mask_dev)
                    rec_Xs.append(combined)

                avg_rec = torch.stack(rec_Xs, dim=0).mean(dim=0).cpu()
                current_filled = avg_rec.detach()

                del model, denoise_fn, best_state
                if device != "cpu":
                    torch.cuda.empty_cache()

        result = current_filled.numpy() * 2.0

        # Paranoia: ensure observed cells are exactly the original values
        # (floating-point ops can introduce tiny drift).
        result[observed_mask_np] = X[observed_mask_np]

        return result.astype(np.float64)