import numpy as np
class SGD:
    def __init__(self, model, learning_rate, lr_decay = True):
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.zero_grad()

    def cpmpute_gradients(self, y_true, lambda_):
        # 反向传播
        delta3 = self.model.y_pred - y_true
        delta2 = delta3.dot(self.model.W3.T) * self.model.activation_derivative(self.model.h2_bias)
        delta1 = delta2[:, :-1].dot(self.model.W2.T) * self.model.activation_derivative(self.model.h1_bias)
        
        self.grad_W1 = self.model.X_bias.T.dot(delta1[:, :-1])   + lambda_ * self.model.W1
        self.grad_W2 = self.model.h1_bias.T.dot(delta2[:, :-1]) + lambda_ * self.model.W2
        self.grad_W3 = self.model.h2_bias.T.dot(delta3)  + lambda_ * self.model.W3

    def update(self, epoch, num_epochs):
        if self.lr_decay: # consine
            self.learning_rate = 0.5 * self.learning_rate * (1 + np.cos(np.pi * epoch / num_epochs))
        self.model.W1 -= self.learning_rate * self.grad_W1 
        self.model.W2 -= self.learning_rate * self.grad_W2
        self.model.W3 -= self.learning_rate * self.grad_W3
    
    def zero_grad(self):
        self.grad_W1 = np.zeros_like(self.model.W1)
        self.grad_W2 = np.zeros_like(self.model.W2)
        self.grad_W3 = np.zeros_like(self.model.W3)