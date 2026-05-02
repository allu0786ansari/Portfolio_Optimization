"""LSTM architecture for next-day return forecasting."""
import torch
import torch.nn as nn


class ReturnLSTM(nn.Module):
    """Multi-layer LSTM with dropout for return forecasting.

    Architecture:
        Input (batch, seq_len, n_features)
        → LSTM layers with dropout between layers
        → Final hidden state
        → Fully-connected head
        → Scalar output (predicted log return)
    """

    def __init__(
        self,
        input_size: int = 10,       # number of feature columns
        hidden_size: int = 64,      # LSTM hidden units
        num_layers: int = 2,        # stacked LSTM layers
        dropout: float = 0.2,       # dropout between LSTM layers
        fc_hidden: int = 32,        # FC head hidden units
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size,) — predicted next-day log return
        """
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, (h_n, _) = self.lstm(x)

        # Take the last layer's final hidden state
        last_hidden = h_n[-1]                   # (batch, hidden_size)
        pred = self.head(last_hidden).squeeze(-1)  # (batch,)
        return pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)