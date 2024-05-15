import numpy as np

def cross_entropy_l2_loss(y_true, y_pred, lambda_, model):
    return -np.mean(y_true * np.log(y_pred + 1e-9)) + 0.5 * lambda_ * (np.mean(model.W1 ** 2) + np.mean(model.W2 ** 2) + np.mean(model.W3 ** 2))

# 计算准确率
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)