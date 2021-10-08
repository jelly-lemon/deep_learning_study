# 版本
Anaconda: 5.2

Python: 3.6

tensorflow: 2.1.4

# 数据集
## MNIST
```python
from tensorflow_core.python.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## IMDB
```python
from tensorflow_core.python.keras.datasets import imdb
imdb.load_data()
```


# 遇到的问题



