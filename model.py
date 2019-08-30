"""
Implements base model class for deep convolutional adversarial network
"""

# def assert_types(obj, name, expectedType):
#     """ Helper to assert proper typing of function inputs """
#     assert isinstance(obj, expectedType), f'{name} expected type {expectedType}, but found type {type{obj}}'

import numpy as np
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.layers import (Input, Conv2D, Activation, LeakyReLU, Dropout,
                            Flatten, Dense, BatchNormalization, ReLU,
                            UpSampling2D, Conv2DTranspose, Reshape)


class GAN(object):

    def __init__(self, name, rowNum, columnNum, channelNum):
        self.name   =   name
        # data formats
        self.rowNum     =   rowNum
        self.columnNum  =   columnNum
        self.channelNum =   channelNum
        self.imageShape =   (rowNum, columnNum, channelNum)
        # model structures
        self.discriminatorStructure =   None
        self.generatorStructure     =   None
        # compiled models
        self.discriminatorCompiled  =   None
        self.adversarialCompiled    =   None
        ## model building params ##
        # default first-layer filter depth of discriminator
        DIS_DEPTH               =   64
        self.DIS_DEPTH          =   DIS_DEPTH
        self.GEN_DEPTH          =   DIS_DEPTH * 4
        # default dropout; should prevent memorization
        self.DROPOUT            =   0.4
        # default kernel size
        self.KERNEL_SIZE        =   5
        # default convolution stride length
        self.STRIDE             =   2
        # default alpha of LeakyReLU activation in discriminator
        self.LEAKY_ALPHA        =   0.2
        # dimensions of generator latent space
        self.LATENT_DIMS        =   100
        # default momentum for adjusting mean and var in generator batch norm
        self.NORM_MOMENTUM      =   0.9

    def dis_get_filter_num(self, LAYER_COUNTER):
        """
        Determines number of filters to use on convolution layer assuming layer
        count starts at 1.
        """
        return (self.DIS_DEPTH * (2 ** (LAYER_COUNTER - 1)))

    def gen_get_filter_num(self, LAYER_COUNTER):
        """
        Determines number of filters to use on transpose convolution layer
        assuming filters were generated by dis_get_filter_num() and layer count
        starts at 1.
        """
        return int(self.GEN_DEPTH / (2 ** LAYER_COUNTER))

    class ModelWarning(Warning):
        # BUG: warning currently raises exception instead of warning
        """ Class for warnings related to model building and compiling """
        pass

    def build_discriminator(self):
        """
        Builds discriminator architecture without compiling model.
        Uses functional API to allow for easy insertion of non-sequential
        elements. If the model has already been build, it is simply returned.
        Input has the shape of a single image as specified during object
        initialization. Convolutional layers have a filter number determined
        by self.dis_get_filter_num(LAYER_COUNTER), use self.STRIDES strides
        for downsampling, and pad to match input shape. LeakyReLU functions
        with self.LEAKY_ALPHA alpha are used to give gradients to inactive
        units and self.DROPOUT dropout is used to prevent overfitting.
        Final output is the probability that the image is real, according to
        a single-node, dense layer with sigmoid activation.
        """
        if self.discriminatorStructure:
            raise self.ModelWarning('Discriminator has already been built.')
            return self.discriminatorStructure
        # set up local vars for building
        INPUT_SHAPE     =   self.imageShape
        KERNEL_SIZE     =   self.KERNEL_SIZE
        STRIDE          =   self.STRIDE
        DROPOUT         =   self.DROPOUT
        LEAKY_ALPHA     =   self.LEAKY_ALPHA
        LAYER_COUNTER   =   1
        ## discriminator architecture ##
        inputs = Input(shape=INPUT_SHAPE, name='inputs')
        # first conv block
        conv_1 = Conv2D(filters=self.dis_get_filter_num(LAYER_COUNTER),
                        kernel_size=KERNEL_SIZE,
                        strides=STRIDE,
                        input_shape=INPUT_SHAPE,
                        padding='same',
                        name=f'conv_{LAYER_COUNTER}')(inputs)
        relu_1 = LeakyReLU(RELU_ALPHA, name=f'relu_{LAYER_COUNTER}')(conv_1)
        drop_1 = Dropout(rate=DROPOUT, name=f'drop_{LAYER_COUNTER}')(relu_1)
        # second conv block
        LAYER_COUNTER += 1
        conv_2 = Conv2D(filters=self.dis_get_filter_num(LAYER_COUNTER),
                        kernel_size=KERNEL_SIZE,
                        strides=STRIDE,
                        input_shape=INPUT_SHAPE,
                        padding='same',
                        name=f'conv_{LAYER_COUNTER}')(drop_1)
        relu_2 = LeakyReLU(RELU_ALPHA, name=f'relu_{LAYER_COUNTER}')(conv_2)
        drop_2 = Dropout(rate=DROPOUT, name=f'drop_{LAYER_COUNTER}')(relu_2)
        # third conv block
        LAYER_COUNTER += 1
        conv_3 = Conv2D(filters=self.dis_get_filter_num(LAYER_COUNTER),
                        kernel_size=KERNEL_SIZE,
                        strides=STRIDE,
                        input_shape=INPUT_SHAPE,
                        padding='same',
                        name=f'conv_{LAYER_COUNTER}')(drop_2)
        relu_3 = LeakyReLU(RELU_ALPHA, name=f'relu_{LAYER_COUNTER}')(conv_3)
        drop_3 = Dropout(rate=DROPOUT, name=f'drop_{LAYER_COUNTER}')(relu_3)
        # fourth conv block
        LAYER_COUNTER += 1
        conv_4 = Conv2D(filters=self.dis_get_filter_num(LAYER_COUNTER),
                        kernel_size=KERNEL_SIZE,
                        strides=STRIDE,
                        input_shape=INPUT_SHAPE,
                        padding='same',
                        name=f'conv_{LAYER_COUNTER}')(drop_3)
        relu_4 = LeakyReLU(RELU_ALPHA, name=f'relu_{LAYER_COUNTER}')(conv_4)
        drop_4 = Dropout(rate=DROPOUT, name=f'drop_{LAYER_COUNTER}')(relu_4)
        # convolutional output is flattened and passed to dense classifier
        flat = Flatten(name='flat')(drop_4)
        outputs = Dense(units=1, activation='sigmoid', name='outputs')(flat)
        # build sequential model
        discriminatorStructure = Model(inputs=inputs, outputs=outputs)
        print(discriminatorStructure.summary())
        self.discriminatorStructure = discriminatorStructure
        return discriminatorStructure

    def build_generator(self):
        """ Builds generator architecture without compiling model """
        if self.generatorStructure:
            raise self.ModelWarning('Generator has already been built.')
            return self.generatorStructure
        # set up local vars for building
        LATENT_DIMS     =   self.LATENT_DIMS
        KERNEL_SIZE     =   self.KERNEL_SIZE
        DROPOUT         =   self.DROPOUT
        NORM_MOMENTUM   =   self.NORM_MOMENTUM
        GEN_DEPTH       =   self.GEN_DEPTH
        # # TEMP: Find out if better params exist
        GEN_DIM         =   7
        LATENT_RESHAPE  =   (GEN_DIM, GEN_DIM, GEN_DEPTH)
        LATENT_NODES    =   GEN_DIM * GEN_DIM * GEN_DEPTH
        LAYER_COUNTER   =   1
        ## generator architecture ##
        latent_inputs = Input(shape=(LATENT_DIMS, ), name='latent_inputs')
        # dense layer to adjust and norm latent space
        dense_latent = Dense(units=LATENT_NODES,
                            input_dim=GEN_INPUT_DIM,
                            name='dense_latent')(latent_inputs)
        batch_latent = BatchNormalization(momentum=NORM_MOMENTUM,
                                        name='batch_latent')(dense_latent)
        relu_latent = ReLU(name='relu_latent')(batch_latent)
        # reshape latent dims into image shape matrix
        reshaped_latent = Reshape(target_shape=LATENT_RESHAPE,
                                name='reshaped_latent')(relu_latent)
        dropout_latent = Dropout(rate=DROPOUT,
                                name='dropout_latent')(reshaped_latent)
        # first upsampling block
        upsample_1 = UpSampling2D(name=f'upsample_{LAYER_COUNTER}')(dropout_latent)
        transpose_1 = Conv2DTranspose(filters=self.gen_get_filter_num(LAYER_COUNTER),
                                    kernel_size=KERNEL_SIZE,
                                    padding='same',
                                    name=f'transpose_{LAYER_COUNTER}')(upsample_1)
        batch_1 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name=f'batch_{LAYER_COUNTER}')(transpose_1)
        relu_1 = ReLU(name=f'relu_{LAYER_COUNTER}')(batch_1)
        # second upsampling block
        LAYER_COUNTER += 1
        upsample_2 = UpSampling2D(name=f'upsample_{LAYER_COUNTER}')(relu_1)
        transpose_2 = Conv2DTranspose(filters=self.gen_get_filter_num(LAYER_COUNTER),
                                    kernel_size=KERNEL_SIZE,
                                    padding='same',
                                    name=f'transpose_{LAYER_COUNTER}')(upsample_2)
        batch_2 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name=f'batch_{LAYER_COUNTER}')(transpose_2)
        relu_2 = ReLU(name=f'relu_{LAYER_COUNTER}')(batch_2)
        # third upsampling block: no upsampling for now
        # QUESTION: Will transpose on final layers lead to artifacts?
        LAYER_COUNTER += 1
        transpose_3 = Conv2DTranspose(filters=self.gen_get_filter_num(LAYER_COUNTER),
                                    kernel_size=KERNEL_SIZE,
                                    padding='same',
                                    name=f'transpose_{LAYER_COUNTER}')(relu_2)
        batch_3 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name=f'batch_{LAYER_COUNTER}')(transpose_3)
        relu_3 = ReLU(name=f'relu_{LAYER_COUNTER}')(batch_3)
        # sigmoid activation on final output to assert grayscale output
        # in range [0, 1]
        output_transpose = Conv2DTranspose(filters=1,
                                            kernel_size=5,
                                            padding='same',
                                            name='output_transpose')(relu_3)
        outputs = Activation(activation='sigmoid')(output_transpose)
        # build sequential model
        generatorStructure = Model(inputs=latent_inputs, outputs=outputs)
        print(generatorStructure.summary())
        self.generatorStructure = generatorStructure
        return generatorStructure

    def compile_discriminator(self):
        """ Compiles discriminator model """
        if self.discriminatorCompiled:
            raise self.ModelWarning('Discriminator has already been compiled.')
            return discriminatorCompiled
        rmsOptimizer = RMSprop(lr=0.0002, decay=6e-8)
        binaryLoss = 'binary_crossentropy'
        discriminatorModel = self.discriminatorStructure
        discriminatorModel.compile(optimizer=rmsOptimizer, loss=binaryLoss,
                                metrics=['accuracy'])
        self.discriminatorCompiled = discriminator
        return discriminator

    def compile_adversarial(self):
        """ Compiles generator model """
        if self.generatorCompiled:
            raise self.ModelWarning('Generator has already been compiled.')
        rmsOptimizer = RMSprop(lr=0.0001, decay=3e-8)
        binaryLoss = 'binary_crossentropy'
        # adversarial built by passing generator output through discriminator
        adversarialModel = Sequential()
        adversarialModel.add(self.discriminatorStructure)
        adversarialModel.add(self.generatorStructure)
        adversarialModel.compile(optimizer=rmsOptimizer, loss=binaryLoss,
                                metrics=['accuracy'])
        self.adversarialCompiled = adversarialModel
        return adversarialModel

    def train_models(self, xTrain, yTrain, xVal, yVal, xTest, yTest,
                    steps, batchSize):
        """
        Trains discriminator, generator, and adversarial model on x- and yTrain,
        validation on x- and yVal and evaluating final metrics on x- and yTest.
        Generator latent space is initialized with random uniform noise between
        -1. and 1.
        Args:
            xTrain:             Training features for discriminator to classify
                                    and generator to 'replicate'
            yTrain:             Labels for training data
            xVal (Optional):    Validation features to analyze training progress
            yVal:               Validation labels to analyze training progress
            xTest:              Test features to analyze model performance
                                    after training
            yTest:              Test labels to analyze model performance after
                                    training
            steps:              Number of steps to take over the data during
                                    model training
            batchSize:          Number of examples over which to compute
                                    gradient during model training
        """

        def shape_assertion(dataset, name):
            """ Asserts that dataset has the proper shape """
            assert (dataset.shape[1:]==self.imageShape), f'{name} expected shape {self.imageShape}, but found shape {dataset.shape}.'

        def length_assertion(dataset_1, dataset_2, name_1, name_2):
            """ Asserts that two datasets have the same example number """
            shape_1 = dataset_1.shape[0]
            shape_2 = dataset_2.shape[0]
            assert (shape_1==shape_2), (f'{name_1} and {name_2} should' \
            f'have the same number of examples, but have {shape_1} and {shape_2}')

        datasetInputs = [('xTrain':xTrain), ('yTrain':yTrain), ('xVal':xVal),
                        ('yVal':yVal), ('xTest':xTest), ('yTest':yTest)]

        for i, (name_1, dataset_1) in enumerate(datasetInputs):
            shape_assertion(dataset_1, name_1)
            if ((i % 2) == 0):
                name_2, dataset_2 = datasetInputs[i+1]
                length_assertion(dataset_1, dataset_2, name_1, name_2)

        assert isinstance(steps, int), f'steps expected type int, but found type {type(steps)}.'
        assert isinstance(batchSize, int), f'batchSize expected type int, but found type {type(batchSize)}'
        assert (self.discriminatorStructure), "Desriminator structure has not been built. Try running 'self.build_discriminator()'."
        assert (self.generatorStructure), "Generator structure has not been built. Try running 'self.build_generator()'."
        assert (self.discriminatorCompiled), "Discriminator model has not been compiled. Try running 'self.compile_discriminator()'."
        assert (self.adversarialCompiled), "Adversarial model has not been compiled. Try running 'self.compile_adversarial()'."


        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
            
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))
