import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class Client:
    """Simple client SDK class for local training.

    - `data` may be a tuple `(X, y)` of torch tensors, a pandas DataFrame, or a path to a partition file.
    - `load_data()` normalizes inputs to `(X_tensor, y_tensor)`.
    - `train_local(global_state, config)` performs a small local training loop and returns a dict
      containing `delta` (state_dict differences), `num_samples`, and `metrics`.
    """

    def __init__(self, client_id: int, data: Optional[Union[str, pd.DataFrame, Tuple[torch.Tensor, torch.Tensor]]] = None):
        self.client_id = int(client_id)
        self._raw_data = data
        self._X = None
        self._y = None

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self._raw_data, tuple):
            X, y = self._raw_data
            self._X = X.clone() if isinstance(X, torch.Tensor) else torch.tensor(X)
            self._y = y.clone() if isinstance(y, torch.Tensor) else torch.tensor(y)
            return self._X, self._y

        if isinstance(self._raw_data, pd.DataFrame):
            df = self._raw_data
            # expect last column to be label
            arr = df.to_numpy()
            X = torch.tensor(arr[:, :-1]).float()
            y = torch.tensor(arr[:, -1]).float()
            self._X, self._y = X, y
            return X, y

        if isinstance(self._raw_data, str):
            path = self._raw_data
            if path.lower().endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            arr = df.to_numpy()
            X = torch.tensor(arr[:, :-1]).float()
            y = torch.tensor(arr[:, -1]).float()
            self._X, self._y = X, y
            return X, y

        raise ValueError("Unsupported or missing data for client. Provide (X,y), DataFrame, or file path.")

    def train_local(self, global_state: dict, config: dict) -> dict:
        model = global_state.get("model")
        if model is None or not isinstance(model, nn.Module):
            raise ValueError("global_state must include a torch.nn.Module under key 'model'")

        X, y = self.load_data()
        dataset = TensorDataset(X, y)
        batch_size = int(config.get("batch_size", 8))
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-3))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # device handling
        device = config.get("device", "cpu")

        # seed control (optional) for reproducibility
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed) + int(self.client_id))

        local_model = copy.deepcopy(model).to(device)

        # loss function: prefer config, then global_state, else MSE
        loss_fn = config.get("loss_fn") or global_state.get("loss_fn") or nn.MSELoss()

        # optimizer: allow configurable optimizer class (defaults to SGD)
        opt_cls = config.get("optimizer", torch.optim.SGD)
        opt_kwargs = config.get("optimizer_kwargs", {})
        opt = opt_cls(local_model.parameters(), lr=lr, **opt_kwargs)

        local_model.train()
        running_loss = 0.0
        seen = 0

        # optional scheduler
        scheduler = None
        sched_conf = config.get("scheduler")
        if sched_conf is not None:
            # sched_conf may be a class or a tuple (class, kwargs)
            if isinstance(sched_conf, tuple) and len(sched_conf) == 2:
                sched_cls, sched_kwargs = sched_conf
            else:
                sched_cls = sched_conf
                sched_kwargs = config.get("scheduler_kwargs", {})
            try:
                scheduler = sched_cls(opt, **(sched_kwargs or {}))
            except Exception:
                scheduler = None

        # early stopping config uses training loss (no val split yet)
        early_cfg = config.get("early_stopping", {})
        use_early = bool(early_cfg.get("enabled", False))
        patience = int(early_cfg.get("patience", 2))
        min_delta = float(early_cfg.get("min_delta", 0.0))
        best_loss = float("inf")
        rounds_no_improve = 0

        epochs_ran = 0
        for epoch in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                preds = local_model(xb).squeeze()
                loss = loss_fn(preds, yb.float())
                loss.backward()
                opt.step()

                running_loss += float(loss.detach().cpu().item()) * xb.size(0)
                seen += xb.size(0)

            # scheduler step per-epoch (support ReduceLROnPlateau via metric)
            if scheduler is not None:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    avg_epoch_loss = float(running_loss / seen) if seen > 0 else 0.0
                    try:
                        scheduler.step(avg_epoch_loss)
                    except Exception:
                        pass
                else:
                    try:
                        scheduler.step()
                    except Exception:
                        pass

            # early stopping check
            epochs_ran += 1
            if use_early:
                avg_epoch_loss = float(running_loss / seen) if seen > 0 else 0.0
                if best_loss - avg_epoch_loss > min_delta:
                    best_loss = avg_epoch_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= patience:
                    break

        avg_loss = float(running_loss / seen) if seen > 0 else 0.0

        # compute delta = local_state - global_state (on CPU)
        delta = {}
        global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        local_sd = {k: v.cpu() for k, v in local_model.state_dict().items()}
        for k in global_sd.keys():
            delta[k] = (local_sd[k] - global_sd[k]).detach().clone()

        return {
            "client_id": self.client_id,
            "num_samples": int(X.size(0)),
            "delta": delta,
            "metrics": {"loss": avg_loss},
            "epochs_ran": int(epochs_ran),
        }
