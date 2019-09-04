from model import DC_GAN
from tensorflow.examples.tutorials.mnist import input_data

# read mnist data from tensorflow database
mnistObj = input_data.read_data_sets("MNIST_data/", one_hot=True)

# pull out data and reshape images into proper shape
# training
xTrain = mnistObj.train.images
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
yTrain = mnistObj.train.labels
# validation
xVal = mnistObj.validation.images
xVal = xVal.reshape(xVal.shape[0], 28, 28, 1)
yVal = mnistObj.validation.labels
# testing
xTest = mnistObj.test.images
xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)
yTest = mnistObj.test.labels

# initialize deep convolutional gan
mnistGAN = DC_GAN(name='mnist_gan', rowNum=28, columnNum=28, channelNum=1)
mnistGAN.initialize_models(disLr=0.0002, advLr=0.00009, verbose=True)
mnistGAN.train_models(xTrain=xTrain, yTrain=yTrain, xVal=xVal, yVal=yVal,
    xTest=xTest, yTest=yTest, trainSteps=80, preSteps=8, batchSize=200,
    saveInterval=5, outPath='outs')
mnistGAN.interpolate(5)
