import torch.nn as nn

class LastLayer(nn.Module):
    def __init__(self, in_features=512, out_features=2):
        super(LastLayer, self).__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)