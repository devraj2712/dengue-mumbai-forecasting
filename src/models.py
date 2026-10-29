import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Computes attention scores over temporal time steps (sequence length).
    Allows the model to dynamically weight which past week (e.g. week t-8 vs week t-4)
    has the strongest predictive impact on the outbreak.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_length, hidden_dim)
        scores = self.attn(rnn_outputs) # (batch_size, seq_length, 1)
        weights = F.softmax(scores, dim=1) # (batch_size, seq_length, 1)
        context = torch.sum(weights * rnn_outputs, dim=1) # (batch_size, hidden_dim)
        return context, weights

class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention for Climate-Informed
    Epidemiological Forecasting.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x) # (batch_size, seq_length, hidden_dim * 2)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context) # (batch_size, 1)
        return out.squeeze(-1), attn_weights

class StandardLSTM(nn.Module):
    """
    Standard Unidirectional LSTM benchmark model.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        # Take the last time step representation
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out.squeeze(-1), None
