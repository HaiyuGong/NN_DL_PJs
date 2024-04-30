import numpy as np

from utils import cross_entropy_l2_loss, accuracy
from SGD import SGD

def train(model, X_train, y_train, X_val, y_val, lambda_, learning_rate, num_epochs, batch_size, lr_decay=False, save_prefix=''):
    num_train = X_train.shape[0]
    num_batches = num_train // batch_size
    optimizer = SGD(model, learning_rate, lr_decay=False)
    best_acc_val = 0
    loss_train_arr, acc_train_arr, loss_val_arr, acc_val_arr = np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)
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
            loss = cross_entropy_l2_loss(y_batch, y_pred, lambda_, model) 
            loss_train_arr[epoch] += loss
            acc_train_arr[epoch] += accuracy(y_batch, y_pred)
            
            # 计算梯度
            optimizer.cpmpute_gradients(y_batch, lambda_)
            # SGD 更新
            optimizer.update(epoch, num_epochs)

        # 在每个 epoch 结束后计算在测试集上的准确率
        y_pred_val = model.forward(X_val)
        loss_val_arr[epoch] = cross_entropy_l2_loss(y_val, y_pred_val, lambda_, model)
        acc_val = accuracy(y_val, y_pred_val)
        acc_val_arr[epoch] = acc_val
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            model.save(f'{save_prefix}_best_model.npz')
    np.savez(f'{save_prefix}_loss_acc.npz', loss_train_arr=loss_train_arr / num_batches, acc_train_arr=acc_train_arr / num_batches, loss_val_arr=loss_val_arr, acc_val_arr=acc_val_arr)
    model.save(f'{save_prefix}_final_model.npz')


