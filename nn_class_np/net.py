import numpy as np

from utils import softmax

class ThreeLayerNN:
    def __init__(self, D_in, H1, H2, D_out ,activation = 'relu'):
        # 初始化权重
        self.W1 = np.random.randn(D_in + 1, H1)
        self.W2 = np.random.randn(H1 + 1, H2)
        self.W3 = np.random.randn(H2 + 1, D_out)
        if activation == 'relu':
            self.activation = lambda x: np.maximum(x, 0)
            self.activation_derivative = lambda x: 1. * (x > 0)  # relu(x) > 0 ? 1 : 0
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: x * (1 - x)  # sigmoid(x) * (1 - sigmoid(x))
        else:
            raise ValueError('Activation function not recognized')
        
    def forward(self, X): 
        # self.X = X 
        # 前向传播
        self.X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        self.h1 = self.activation(np.dot(self.X_bias, self.W1))
        self.h1_bias = np.hstack([self.h1, np.ones((self.h1.shape[0], 1))])
        self.h2 = self.activation(np.dot(self.h1_bias, self.W2))
        self.h2_bias = np.hstack([self.h2, np.ones((self.h2.shape[0], 1))])
        self.y_pred = softmax(np.dot(self.h2_bias, self.W3))
        return self.y_pred  
    
    def save(self, path):
        np.savez(path, W1=self.W1, W2=self.W2, W3=self.W3)
    
    def load(self, path):
        npzfile = np.load(path)
        self.W1 = npzfile['W1']
        self.W2 = npzfile['W2']
        self.W3 = npzfile['W3']