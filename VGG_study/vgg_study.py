from tensorflow.keras.applications import vgg16, vgg19
from tensorflow import keras



def test_0():
    """
    include_top：是否要最后的 3 层全连接层
    weights: Node 随机初始化，'imagenet' 下载从 ImageNet 训练的权重
    """
    vgg_16 = vgg16.VGG16(input_shape=(224, 224, 3),
                         weights=None,
                         include_top=False,
                         backend=keras.backend,
                         layers=keras.layers,
                         models=keras.models,
                         utils=keras.utils)
    # vgg_16 = Model()
    print(type(vgg_16))
    print(vgg_16.summary())

def test_1():
    """
    VGG19
    """
    vgg_19 = vgg19.VGG19(input_shape=(224, 224, 3),
                         weights=None,
                         include_top=False,
                         backend=keras.backend,
                         layers=keras.layers,
                         models=keras.models,
                         utils=keras.utils)
    print(type(vgg_19))
    print(vgg_19.summary())


if __name__ == '__main__':
    test_1()