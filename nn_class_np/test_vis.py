import numpy as np
import matplotlib.pyplot as plt

from net import ThreeLayerNN
import mnist_reader
from train import train
from utils import cross_entropy_l2_loss, accuracy


def test(model, X_test, y_test, lambda_, save_prefix):
    model.load(f'{save_prefix}_best_model.npz')
    y_pred_val = model.forward(X_test)
    loss_test = cross_entropy_l2_loss(y_test, y_pred_val, lambda_, model)
    acc_test = accuracy(y_test, y_pred_val)
    print(f'Test loss: {loss_test:.4f}, Test accuracy: {acc_test:.4f}')

def plot_loss_acc(save_prefix):
    npzfile = np.load(f'{save_prefix}_loss_acc.npz')
    loss_train_arr, acc_train_arr, loss_val_arr, acc_val_arr = npzfile['loss_train_arr'], npzfile['acc_train_arr'], npzfile['loss_val_arr'], npzfile['acc_val_arr']
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(loss_train_arr, label='train')
    plt.plot(loss_val_arr, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(acc_train_arr, label='train')
    plt.plot(acc_val_arr, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f'{save_prefix}_loss_acc.png')
    # plt.show()
    plt.close()



if __name__ == '__main__':
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

    model = ThreeLayerNN(784, 100, 100, 10, activation='sigmoid')
    test(model, X_test, y_test, lambda_=0.001, save_prefix='weights/0idx_200epochs_100H1_100H2_0.001lambda_0.002lr_Falselr_decay')
    plot_loss_acc('weights/0idx_200epochs_100H1_100H2_0.001lambda_0.002lr_Falselr_decay')