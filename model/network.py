import numpy as np

class SiloSpanClassifier:
    def __init__(self, input_dim: int = 8, hidden_dims: list = [16, 8], output_dim: int = 2, dropout: float = 0.0):
        """
        A parameterizable Multi-Layer Perceptron (MLP) Classifier implemented in pure NumPy.
        Mimics PyTorch linear layers and ReLUs for compatibility.
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Initialize weights and biases (using Kaiming/He normal initialization)
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # He normal initialization
            w = np.random.randn(h_dim, prev_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(h_dim)
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))
            prev_dim = h_dim
            
        # Final output layer
        w = np.random.randn(output_dim, prev_dim) * np.sqrt(2.0 / prev_dim)
        b = np.zeros(output_dim)
        self.weights.append(w.astype(np.float32))
        self.biases.append(b.astype(np.float32))
        
        self.training = True
        self.activations = []
        self.zs = []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        # We need to cache activations and zs for backpropagation
        self.activations = [x]
        self.zs = []
        
        a = x
        num_layers = len(self.weights)
        for i in range(num_layers - 1):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(a, w.T) + b
            self.zs.append(z)
            a = np.maximum(0, z)  # ReLU
            
            if self.dropout > 0 and self.training:
                # Inverted dropout mask
                mask = (np.random.rand(*a.shape) >= self.dropout) / (1.0 - self.dropout)
                a = a * mask
            self.activations.append(a)
            
        # Final output layer (no ReLU, raw logits)
        w = self.weights[-1]
        b = self.biases[-1]
        z = np.dot(a, w.T) + b
        self.zs.append(z)
        return z

    def backward(self, x, y, lr=0.01):
        # Softmax computation on the cached final layer logits
        logits = self.zs[-1]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        n = x.shape[0]
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), y] = 1.0
        
        # Derivative of Loss w.r.t logits (CrossEntropy + Softmax)
        dzo = (probs - y_onehot) / n
        
        num_layers = len(self.weights)
        dz = dzo
        
        for i in reversed(range(num_layers)):
            a_prev = self.activations[i]
            w = self.weights[i]
            
            # Gradients of loss w.r.t weights and biases for layer i
            dw = np.dot(dz.T, a_prev)
            db = np.sum(dz, axis=0)
            
            # Update weights and biases
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            
            if i > 0:
                # Backpropagate gradient to previous layer activation
                da_prev = np.dot(dz, w)
                z_prev = self.zs[i-1]
                # Gradient of ReLU
                dz = da_prev * (z_prev > 0)
