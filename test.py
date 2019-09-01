import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnistObj = input_data.read_data_sets("MNIST_data/", one_hot=True)
xTrain = mnistObj.train.images
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
yTrain = mnistObj.train.labels
for i in range(10):
    img = xTrain[i]
    img = img[:,:,0]
    assert img.shape==(28,28), f'{img.shape}'
    print(f'Target: {yTrain[i]}')
    plt.imshow(img)
    plt.show()
