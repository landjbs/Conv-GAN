import keras
# input is (x_pixels, y_pixels, channels), batch_size dropped
INPUT_SHAPE = features.shape[:1]
# filter depth of the model
DEPTH = 64
# dropout of the model; high dropout should prevent memorization in generator
DROPOUT = 0.4
# set default kernel_size
KERNEL_SIZE = 5
# set default stride length
STRIDE = 2
# set default alpha param of leaky ReLU activation across layers
RELU_ALPHA = 0.2


# convolutional layers have ((2**i) * DEPTH) filter number for i in range(0,4),
# use strides of 2 for downsampling, and pad to match input shape. Leaky ReLU
# activation functions are used to give gradients to inactive units
inputs = keras.layers.Input(shape=INPUT_SHAPE)
# first conv block
conv_1 = keras.layers.Conv2D(filters=(DEPTH*1), kernel_size=KERNEL_SIZE,
                            strides=STRIDE, input_shape=INPUT_SHAPE,
                            activation=keras.layers.LeakyReLU(RELU_ALPHA),
                            padding='same')(input)
drop_1 = keras.layers.Dropout(DROPOUT)(conv_1)
# second conv block
conv_2 = keras.layers.Conv2D(filters=(DEPTH*2), kernel_size=KERNEL_SIZE,
                            strides=STRIDE,
                            activation=keras.layers.LeakyReLU(RELU_ALPHA),
                            padding='same')(drop_1)
drop_2 = keras.layers.Dropout(DROPOUT)(conv_2)
# third conv block
conv_3 = keras.layers.Conv2D(filters=(DEPTH*4), kernel_size=KERNEL_SIZE,
                            strides=STRIDE,
                            activation=keras.layers.LeakyReLU(RELU_ALPHA),
                            padding='same')(drop_2)
drop_3 = keras.layers.Dropout(DROPOUT)(conv_3)
# fourth conv block
conv_4 = keras.layers.Conv2D(filters=(DEPTH*8), kernel_size=KERNEL_SIZE,
                            strides=STRIDE,
                            activation=keras.layers.LeakyReLU(RELU_ALPHA),
                            padding='same')(drop_3)
drop_4 = keras.layers.Dropout(DROPOUT)(conv_4)
# convolutional output is flattened and passed to dense layer for classification
flat = keras.layers.Flatten()(drop_4)
outputs = keras.layers.Dense(units=1, activation='sigmoid')(flat)
# build sequential model
discriminator = keras.models.Model(inputs=inputs, outputs=outputs)
