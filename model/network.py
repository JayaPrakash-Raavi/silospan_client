import torch
import torch.nn as nn
import torch.nn.functional as F

class SiloSpanClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dims: list = [16, 8], output_dim: int = 2, dropout: float = 0.0):
        """
        A parameterizable Multi-Layer Perceptron (MLP) Classifier.
        Allows setting custom input dimensions, list of hidden layers, output classes, and dropout.
        """
        super(SiloSpanClassifier, self).__init__()
        layers = []
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
