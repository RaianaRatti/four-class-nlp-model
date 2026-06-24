import torch
import torch.nn as nn

INPUT_DIM   = 128
HIDDEN_DIM  = 512
LSTM_HIDDEN = 128
CONTEXT_FRAMES = 7   # how many frames of history the model sees (7 × 30ms = 210ms)
NUM_CLASSES = 4


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class VADNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Same frame-level feature extractor as before
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(HIDDEN_DIM, dropout=0.3),
            ResBlock(HIDDEN_DIM, dropout=0.3),
        )

        # NEW: LSTM reads a sequence of frame embeddings and outputs
        # a context-aware representation for each frame
        self.lstm = nn.LSTM(
            input_size=HIDDEN_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,        # expects (batch, seq_len, features)
            bidirectional=False,     # causal — only looks at past frames
        )

        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, INPUT_DIM) during training
        #          (batch, INPUT_DIM) during single-frame inference — handled below

        single_frame = x.ndim == 2
        if single_frame:
            x = x.unsqueeze(1)          # (batch, 1, INPUT_DIM)

        batch, seq_len, _ = x.shape

        # Apply frame-level encoder to every frame in the sequence
        x = x.view(batch * seq_len, INPUT_DIM)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = x.view(batch, seq_len, HIDDEN_DIM)

        # LSTM over the sequence
        x, _ = self.lstm(x)             # (batch, seq_len, LSTM_HIDDEN)

        # Classify each frame using its context-aware representation
        x = self.head(x)                # (batch, seq_len, NUM_CLASSES)

        if single_frame:
            x = x.squeeze(1)            # back to (batch, NUM_CLASSES)

        return x
    
'''

Notes:
    1. class VADNet(nn.Module) --> creates class that inherits from nn.Module which contains basic neural network capabilities

    2. super().__init__() --> initializes parent class (nn.Module) which needs to be initialized so child class (VADNet) can use its capabilities

    3. self.net --> creates variable belonging to VADNet class (called net)

    4. nn.Sequential --> Connects the layers provided inside of it in order --> 
    
        - nn.Sequential(A, B, C) 
        - roughly translates to
        - def forward(x):
            x = A(x)
            x = B(x)
            x = C(x)
            return x
    
    5. Linear --> output = W * input + b (W = weights, b = bias, both learned during training)

    6. ReLU --> if x < 0: x = 0
        - Why: Only doing linear layers consecutively acts as a giant linear layer, ReLU introduces non-linearity which helps model learn some complex relationships

    7. LayerNorm --> output = output that is scaled to stay within std of 1
        - Why: Prevents huge discrepancy in numbers between two samples (Ex: [1000, 1000, 1013] and [0.01, 0.02, 0.03])

    8. Dropout --> randomly disables 30% of neurons' outputs (setting them to 0)
        - Why: Without this, nn might memorize training data

'''