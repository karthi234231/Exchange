"""
models/dl_models.py — Deep learning model architectures for NSE 15-min bar trading.

Models:
  MLP        - sklearn MLPClassifier (works without PyTorch; 2-3 hidden layers)
  LSTMModel  - PyTorch LSTM sequence model (requires torch)
  TCNModel   - Temporal Convolutional Network (requires torch)
  CNNLSTMModel - 1D-CNN feature extractor + LSTM (requires torch)

All models operate on:
  - Tabular input (MLP, XGB, LGB): shape (n_samples, n_features)
  - Sequential input (LSTM, TCN, CNN-LSTM): shape (n_samples, seq_len, n_features)

Label convention: {-1, 0, +1} → {0, 1, 2} (SELL / HOLD / BUY)
"""
from __future__ import annotations

from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# MLP (sklearn) — works without PyTorch
# ---------------------------------------------------------------------------

class SklearnMLP:
    """Multilayer Perceptron classifier via sklearn.

    Architecture:
        Input(n_features)
        Dense(256, relu) → BatchNorm → Dropout
        Dense(128, relu) → Dropout
        Dense(64, relu)
        Dense(3, softmax)

    Args:
        hidden_sizes: Tuple of hidden layer sizes.
        dropout: Not directly supported by sklearn MLP; use alpha for regularisation.
        alpha: L2 regularisation strength.
        max_iter: Maximum training epochs.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        alpha: float = 1e-4,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> None:
        from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=1024,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
            verbose=False,
        )
        self._label_map = {-1: 0, 0: 1, 1: 2}
        self._inv_map = {0: -1, 1: 0, 2: 1}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnMLP":
        X_scaled = self._scaler.fit_transform(X)
        y_mapped = np.array([self._label_map.get(int(v), 1) for v in y])
        self._model.fit(X_scaled, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        mapped = self._model.predict(X_scaled)
        return np.array([self._inv_map[int(v)] for v in mapped])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probabilities for classes {0=SELL, 1=HOLD, 2=BUY}."""
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    @property
    def n_iter_(self) -> int:
        return getattr(self._model, "n_iter_", 0)


# ---------------------------------------------------------------------------
# PyTorch LSTM — conditional on torch being installed
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class LSTMModel:
    """Bidirectional LSTM sequence classifier.

    Architecture:
        Input: (batch, seq_len, n_features)
        BiLSTM(hidden=128, layers=2, dropout=0.3)
        Dense(64, relu) → Dropout(0.3)
        Dense(3, softmax)

    Args:
        n_features: Number of input features per time step.
        seq_len: Input sequence length (number of 15-min bars).
        hidden_size: LSTM hidden state size.
        n_layers: Number of LSTM layers.
        dropout: Dropout rate between LSTM layers.
        lr: Adam learning rate.
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 60,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 128,
    ) -> None:
        if not _torch_available():
            raise ImportError("PyTorch is required for LSTMModel. Install: pip install torch")

        import torch
        import torch.nn as nn

        self.n_features = n_features
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self._label_map = {-1: 0, 0: 1, 1: 2}
        self._inv_map = {0: -1, 1: 0, 2: 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        class _Net(nn.Module):
            def __init__(self, n_feat: int, hidden: int, layers: int, drop: float) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_feat,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=drop if layers > 1 else 0.0,
                    batch_first=True,
                    bidirectional=True,
                )
                self.dropout = nn.Dropout(drop)
                self.fc1 = nn.Linear(hidden * 2, 64)  # *2 for bidirectional
                self.fc2 = nn.Linear(64, 3)
                self.relu = nn.ReLU()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.lstm(x)
                out = out[:, -1, :]  # last timestep
                out = self.dropout(self.relu(self.fc1(out)))
                return self.fc2(out)

        self._net = _Net(n_features, hidden_size, n_layers, dropout).to(self.device)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._criterion = nn.CrossEntropyLoss()

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None) -> tuple:
        """Create sliding window sequences from flat feature matrix."""
        import torch
        n = len(X)
        seqs, labels = [], []
        for i in range(self.seq_len, n):
            seqs.append(X[i - self.seq_len:i])
            if y is not None:
                labels.append(self._label_map.get(int(y[i - 1]), 1))
        X_seq = torch.tensor(np.array(seqs), dtype=torch.float32)
        if y is not None:
            y_seq = torch.tensor(labels, dtype=torch.long)
            return X_seq, y_seq
        return X_seq, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        import torch
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_seq, y_seq = self._make_sequences(X_scaled, y)

        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self._net.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self._optimizer.zero_grad()
                logits = self._net(xb)
                loss = self._criterion(logits, yb)
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(loader)
                print(f"    LSTM epoch {epoch+1}/{self.n_epochs}  loss={avg:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X_scaled = self._scaler.transform(X)
        X_seq, _ = self._make_sequences(X_scaled, None)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(X_seq.to(self.device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        # Pad seq_len rows at the front with uniform probs
        pad = np.full((self.seq_len, 3), 1.0 / 3.0)
        return np.vstack([pad, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        mapped = np.argmax(probs, axis=1)
        return np.array([self._inv_map[int(v)] for v in mapped])


# ---------------------------------------------------------------------------
# PyTorch TCN (Temporal Convolutional Network)
# ---------------------------------------------------------------------------

class TCNModel:
    """Temporal Convolutional Network classifier.

    Architecture:
        Input: (batch, seq_len, n_features)
        Permute → (batch, n_features, seq_len)
        TCN residual blocks: dilation=1,2,4,8 (exponential)
        Global avg pooling
        Dense(64, relu) → Dense(3, softmax)

    Args:
        n_features: Number of input features per time step.
        seq_len: Input sequence length.
        n_channels: Number of channels in each TCN block.
        kernel_size: Convolution kernel size.
        n_blocks: Number of TCN residual blocks.
        dropout: Dropout in residual connections.
        lr: Learning rate.
        n_epochs: Training epochs.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 60,
        n_channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 128,
    ) -> None:
        if not _torch_available():
            raise ImportError("PyTorch is required for TCNModel. Install: pip install torch")

        import torch
        import torch.nn as nn

        self.n_features = n_features
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._label_map = {-1: 0, 0: 1, 1: 2}
        self._inv_map = {0: -1, 1: 0, 2: 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _ResBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, k: int, dil: int, drop: float) -> None:
                super().__init__()
                pad = (k - 1) * dil
                self.conv1 = nn.Conv1d(in_ch, out_ch, k, dilation=dil, padding=pad)
                self.conv2 = nn.Conv1d(out_ch, out_ch, k, dilation=dil, padding=pad)
                self.bn1 = nn.BatchNorm1d(out_ch)
                self.bn2 = nn.BatchNorm1d(out_ch)
                self.drop = nn.Dropout(drop)
                self.relu = nn.ReLU()
                self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # Causal: trim the extra padding from the right
                o = self.relu(self.bn1(self.conv1(x)))
                o = o[:, :, :x.shape[-1]]
                o = self.drop(o)
                o = self.relu(self.bn2(self.conv2(o)))
                o = o[:, :, :x.shape[-1]]
                o = self.drop(o)
                return self.relu(o + self.skip(x))

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList()
                in_ch = n_features
                for i in range(n_blocks):
                    dil = 2 ** i
                    self.blocks.append(_ResBlock(in_ch, n_channels, kernel_size, dil, dropout))
                    in_ch = n_channels
                self.fc1 = nn.Linear(n_channels, 64)
                self.fc2 = nn.Linear(64, 3)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(dropout)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: (batch, seq, feat) → permute to (batch, feat, seq)
                x = x.permute(0, 2, 1)
                for block in self.blocks:
                    x = block(x)
                x = x.mean(dim=-1)  # global avg pool
                x = self.drop(self.relu(self.fc1(x)))
                return self.fc2(x)

        self._net = _Net().to(self.device)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-4)
        self._criterion = nn.CrossEntropyLoss()

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None) -> tuple:
        import torch
        n = len(X)
        seqs, labels = [], []
        for i in range(self.seq_len, n):
            seqs.append(X[i - self.seq_len:i])
            if y is not None:
                labels.append(self._label_map.get(int(y[i - 1]), 1))
        X_seq = torch.tensor(np.array(seqs), dtype=torch.float32)
        if y is not None:
            return X_seq, torch.tensor(labels, dtype=torch.long)
        return X_seq, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNModel":
        import torch
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_seq, y_seq = self._make_sequences(X_scaled, y)

        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self._net.train()
        for epoch in range(self.n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self._optimizer.zero_grad()
                loss = self._criterion(self._net(xb), yb)
                loss.backward()
                self._optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X_scaled = self._scaler.transform(X)
        X_seq, _ = self._make_sequences(X_scaled, None)
        self._net.eval()
        with torch.no_grad():
            probs = F.softmax(self._net(X_seq.to(self.device)), dim=-1).cpu().numpy()
        pad = np.full((self.seq_len, 3), 1.0 / 3.0)
        return np.vstack([pad, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        mapped = np.argmax(probs, axis=1)
        return np.array([self._inv_map[int(v)] for v in mapped])


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return list of model names available in the current environment."""
    models = ["MLP (sklearn)"]
    if _torch_available():
        models += ["LSTM (PyTorch)", "TCN (PyTorch)", "CNN-LSTM (PyTorch)"]
    return models
