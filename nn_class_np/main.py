import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product

from net import ThreeLayerNN
import mnist_reader
from train import train
from test_vis import test, plot_loss_acc

num_epochs_list = [100, 200, 400]
H1_list = [50, 100, 150]
H2_list = [50, 100, 150]
lambda_list = [0.001, 0.01, 0.1]
lr_list = [0.0001, 0.001, 0.01]
lr_decay_list = [True, False]

X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

for idx, (num_epochs, H1, H2, lambda_, learning_rate, lr_decay) in enumerate(product(num_epochs_list, H1_list, H2_list, lambda_list, lr_list, lr_decay_list)):
    print(f'Experiment {idx + 1}/{len(num_epochs_list) * len(H1_list) * len(H2_list) * len(lambda_list) * len(lr_list) * len(lr_decay_list)}, \
    num_epochs={num_epochs}, H1={H1}, H2={H2}, lambda_={lambda_}, learning_rate={learning_rate}, lr_decay={lr_decay}')
    save_prefix = f'weights/{idx}idx_{num_epochs}epochs_{H1}H1_{H2}H2_{lambda_}lambda_{learning_rate}lr_{lr_decay}lr_decay'
    model = ThreeLayerNN(784, H1, H2, 10, activation='sigmoid')
    train(model, X_train, y_train, X_val, y_val, lambda_=lambda_, learning_rate=learning_rate, num_epochs=num_epochs, batch_size=512, lr_decay=lr_decay, save_prefix=save_prefix)
    # test(model, X_test, y_test, lambda_=0.001, save_prefix='weights/0idx_200epochs_100H1_100H2_0.001lambda_0.002lr_Falselr_decay')
    test(model, X_test, y_test, lambda_=lambda_, save_prefix=save_prefix)
    # plot_loss_acc('weights/0idx_200epochs_100H1_100H2_0.001lambda_0.002lr_Falselr_decay')
    plot_loss_acc(save_prefix)
    # break

# model = ThreeLayerNN(784, 100, 100, 10, activation='sigmoid')
# train(model, X_train, y_train, X_val, y_val, lambda_=0.001, learning_rate=0.005, num_epochs=200, batch_size=512, lr_decay=False, save_prefix='weights/0idx_200epochs_100H1_100H2_0.001lambda_0.002lr_Falselr_decay')