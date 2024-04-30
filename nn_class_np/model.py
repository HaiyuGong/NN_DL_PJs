import numpy as np
from numpy.random import randn

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9))

def compute_gradients(model, X, y_true, lambda_):
    # 反向传播
    delta3 = model.y_pred - y_true
    delta2 = delta3.dot(model.W3.T) * model.activation_derivative(model.h2_bias)
    delta1 = delta2[:, :-1].dot(model.W2.T) * model.activation_derivative(model.h1_bias)
    
    grad_W1 = X.T.dot(delta1) + lambda_ * model.W1
    grad_W2 = model.h1_bias.T.dot(delta2[:, :-1]) + lambda_ * model.W2
    grad_W3 = model.h2_bias.T.dot(delta3) + lambda_ * model.W3
    
    return grad_W1, grad_W2, grad_W3

def sgd_update(model, grad_W1, grad_W2, grad_W3, step_size):
    # SGD更新
    model.W1 -= step_size * grad_W1
    model.W2 -= step_size * grad_W2
    model.W3 -= step_size * grad_W3


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
        # 前向传播
        self.X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        self.h1 = self.activation(np.dot(self.X_bias, self.W1))
        self.h1_bias = np.hstack([self.h1, np.ones((self.h1.shape[0], 1))])
        self.h2 = self.activation(np.dot(self.h1_bias, self.W2))
        self.h2_bias = np.hstack([self.h2, np.ones((self.h2.shape[0], 1))])
        self.y_pred = softmax(np.dot(self.h2_bias, self.W3))
        return self.y_pred  

# 计算准确率
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def compute_gradients(model, y_true, lambda_):
    # 反向传播
    delta3 = model.y_pred - y_true
    delta2 = delta3.dot(model.W3.T) * model.activation_derivative(model.h2_bias)
    delta1 = delta2[:, :-1].dot(model.W2.T) * model.activation_derivative(model.h1_bias)
    
    grad_W1 = model.X_bias.T.dot(delta1[:, :-1]) 
    grad_W2 = model.h1_bias.T.dot(delta2[:, :-1]) 
    grad_W3 = model.h2_bias.T.dot(delta3) 
    
    return grad_W1, grad_W2, grad_W3

def sgd_update(model, grad_W1, grad_W2, grad_W3, step_size):
    # SGD更新
    model.W1 -= step_size * grad_W1
    model.W2 -= step_size * grad_W2
    model.W3 -= step_size * grad_W3


def train(model, X_train, y_train, X_test, y_test, lambda_, learning_rate, num_epochs, batch_size):
    num_train = X_train.shape[0]
    num_batches = num_train // batch_size

    for epoch in range(num_epochs):
        # 打乱训练数据集
        permuted_indices = np.random.permutation(num_train)
        X_train_permuted = X_train[permuted_indices]
        y_train_permuted = y_train[permuted_indices]

        for i in range(num_batches):
            # 获取当前批次的数据
            start = i * batch_size
            end = (i + 1) * batch_size
            X_batch = X_train_permuted[start:end]
            y_batch = y_train_permuted[start:end]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失函数
            pred_loss = cross_entropy_loss(y_batch, y_pred) 
            reg_loss = 0.5 * lambda_ * (np.mean(model.W1 ** 2) + np.mean(model.W2 ** 2) + np.mean(model.W3 ** 2))
            reg_loss =0
            loss = pred_loss + reg_loss
            # 计算梯度
            grad_W1, grad_W2, grad_W3 = compute_gradients(model, y_batch, lambda_)

            # SGD 更新
            sgd_update(model, grad_W1, grad_W2, grad_W3, learning_rate)

        # 在每个 epoch 结束后计算在测试集上的准确率
        acc_train = accuracy(y_train, model.forward(X_train))
        y_pred_test = model.forward(X_test)
        acc = accuracy(y_test, y_pred_test)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Train Accuracy: {acc_train:.2%}, Test Accuracy: {acc:.2%}')

        


import mnist_reader
X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

model = ThreeLayerNN(784, 100, 100, 10, activation='relu')
train(model, X_train, y_train_onehot, X_test, y_test_onehot, lambda_=0.0001, learning_rate=0.1, num_epochs=20, batch_size=128)