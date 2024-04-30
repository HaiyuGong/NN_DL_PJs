import numpy as np


from net import ThreeLayerNN
import mnist_reader




# def compute_gradients(model, y_true, lambda_):
#     # 反向传播
#     delta3 = model.y_pred - y_true
#     delta2 = delta3.dot(model.W3.T) * model.activation_derivative(model.h2_bias)
#     delta1 = delta2[:, :-1].dot(model.W2.T) * model.activation_derivative(model.h1_bias)
    
#     grad_W1 = model.X_bias.T.dot(delta1[:, :-1])   + lambda_ * model.W1
#     grad_W2 = model.h1_bias.T.dot(delta2[:, :-1]) + lambda_ * model.W2
#     grad_W3 = model.h2_bias.T.dot(delta3)  + lambda_ * model.W3
    
#     return grad_W1, grad_W2, grad_W3

# def sgd_update(model, epoch, num_epochs, grad_W1, grad_W2, grad_W3, learning_rate, lr_decay = True):
#     # SGD更新
#     if lr_decay: # consine
#         learning_rate = 0.5 * learning_rate * (1 + np.cos(np.pi * epoch / num_epochs))
#     model.W1 -= learning_rate * grad_W1 
#     model.W2 -= learning_rate * grad_W2
#     model.W3 -= learning_rate * grad_W3


