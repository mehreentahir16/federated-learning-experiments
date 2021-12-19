from torch import nn
import torch.nn.functional as F

# Mnist & fmnist 
# CNN 1,  1,663,370 total parameters.
class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x
