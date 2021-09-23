import numpy as np

def load_mnist_data():
    """
    加载 mnist 数据
    """
    with np.load("./mnist.npz", allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)