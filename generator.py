import keras

DROPOUT = 0.4
GEN_DEPTH = 4 * DIS_DEPTH

# dimensions of the latent space
GEN_INPUT_DIM = 100
GEN_DIM = 7
DROPOUT = 0.4
KERNEL_SIZE = 5

# set default momentum for adjusting mean and var in batch normalization
NORM_MOMENTUM = 0.9

get_trans_filter_num = lambda depthMult : int(GEN_DEPTH / depthMult)

latent = keras.layers.Input(shape=(GEN_INPUT_DIM, ), name='latent')

# dense layer to adjust latent space
dense_latent = keras.layers.Dense(units=((GEN_DIM**2) * GEN_DEPTH),
                        input_dim=GEN_INPUT_DIM,
                        name='latent_dense')(latent)
batch_dense = keras.layers.BatchNormalization(momentum=NORM_MOMENTUM)(dense_latent)
relu_dense = keras.layers.Activation(activation=keras.layers.ReLU())(batch_dense)

# reshape weighted latent dims into picture matrix
latent_reshaped = keras.layers.Reshape(target_shape=(GEN_DIM, GEN_DIM, GEN_DEPTH),
                                    name='latent_reshaped')(relu_dense)
latent_dropout = keras.layers.Dropout(DROPOUT)(latent_reshaped)

# upsample latent dims
upsample_1 = keras.layers.UpSampling2D()(latent_dropout)
# transpose convolution on upsampled latent dimensions
transpose_1 = keras.layers.Conv2DTranspose(filters=get_trans_filter_num(1),
                                        kernel_size=KERNEL_SIZE,
                                        padding='same',
                                        name='transpose_1')(upsample_1)
