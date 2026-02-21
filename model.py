import torch
import torch.nn as nn


NUM_CLASSES = 6  # walking, jogging, running, boxing, handwaving, handclapping


# ─────────────────────────── Simple CNN Encoder ───────────────────────────────

class CNN(nn.Module):
    """
    A straightforward 3-block convolutional encoder.

    Block structure: Conv2d → BatchNorm → ReLU → MaxPool

    Input : (B*T, 3, H, W)   e.g. (B*T, 3, 112, 112)
    Output: (B*T, feature_dim)
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim

        # Block 1: 3 → 32 channels,  112×112 → 56×56
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32 → 64 channels,  56×56 → 28×28
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 64 → 128 channels,  28×28 → 14×14
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After 3 poolings on 112×112: spatial size = 14×14
        # Flattened: 128 * 14 * 14 = 25088
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128 * 14 * 14, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B*T, 3, H, W)
        Returns:
            features : (B*T, feature_dim)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ─────────────────────────── Simple Vanilla RNN ───────────────────────────────

class RNN(nn.Module):
    """
    A plain vanilla RNN using PyTorch's nn.RNN (tanh activation).
    No LSTM gates, no GRU gates — just the basic recurrence:

        h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)

    Input : (B, T, input_dim)
    Output: (B, hidden_dim)   ← final hidden state h_T
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",   # vanilla RNN — no gating mechanism
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, T, input_dim)
        Returns:
            h_T : (B, hidden_dim)  — last time-step hidden state
        """
        # rnn_out : (B, T, hidden_dim)  — outputs at every time step
        # h_n     : (num_layers, B, hidden_dim) — final hidden state
        rnn_out, h_n = self.rnn(x)

        # We only need the last layer's hidden state at the final time step
        h_T = h_n[-1]   # (B, hidden_dim)
        return h_T


# ─────────────────────────── Full Model ───────────────────────────────────────

class CNNRNN(nn.Module):
    """
    Simple CNN + Simple Vanilla RNN for video action classification.

    Input : (B, T, C, H, W)
    Output: (B, num_classes)  — raw logits
    """

    def __init__(
        self,
        num_classes:    int = NUM_CLASSES,
        feature_dim:    int = 256,   # CNN output dim  (= RNN input dim)
        hidden_dim:     int = 256,   # RNN hidden state dim
        num_rnn_layers: int = 1,     # single-layer RNN keeps it simple
    ):
        super().__init__()

        # ── Simple CNN ────────────────────────────────────────────────────────
        self.cnn = CNN(feature_dim=feature_dim)

        # ── Simple Vanilla RNN ────────────────────────────────────────────────
        self.rnn = RNN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_rnn_layers,
        )

        # ── Linear Classifier ─────────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C, H, W)
        Returns:
            logits : (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # ── 1. Extract per-frame spatial features with CNN ────────────────────
        x = x.view(B * T, C, H, W)        # flatten batch + time: (B*T, C, H, W)
        features = self.cnn(x)             # (B*T, feature_dim)
        features = features.view(B, T, -1) # restore sequence:    (B, T, feature_dim)

        # ── 2. Model temporal dynamics with RNN ───────────────────────────────
        h_T = self.rnn(features)           # (B, hidden_dim)

        # ── 3. Classify from the final hidden state ───────────────────────────
        logits = self.classifier(h_T)      # (B, num_classes)
        return logits


# ─────────────────────── Utility ──────────────────────────────────────────────

def build_model() -> CNNRNN:
    """Returns the model with default hyperparameters."""
    return CNNRNN(
        num_classes=NUM_CLASSES,
        feature_dim=256,
        hidden_dim=256,
        num_rnn_layers=1,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────── Sanity Check ─────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Forward pass with a dummy batch
    B, T, C, H, W = 2, 16, 3, 112, 112
    dummy = torch.randn(B, T, C, H, W)
    logits = model(dummy)
    print(f"\nInput  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(logits.shape)}  (expected: ({B}, {NUM_CLASSES}))")