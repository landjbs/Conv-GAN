DROPOUT = 0.4
GEN_DEPTH = 4 * DIS_DEPTH

GEN_INPUT_DIM = 100
GEN_DIM = 7
# set default momentum for adjusting mean and var in batch normalization
NORM_MOMENTUM = 0.9

inputs = keras.layers.Input(shape=(GEN_INPUT_DIM, ), name='inputs')
