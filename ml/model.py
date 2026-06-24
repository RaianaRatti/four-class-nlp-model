import torch.nn as nn

INPUT_DIM   = 128  # 40 MFCC + 40 delta + 40 delta2 + energy + ZCR + flatness + centroid + rolloff + voiced_frac + spectral_entropy + harmonic_ratio
HIDDEN_DIM  = 512
NUM_CLASSES = 4    # from dataset.py -> four classes (silence, speech, overlap, vocalization)


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
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.head(x)  # raw logits, shape (batch, 4)
    
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