"""
Implements base model class for deep convolutional adversarial network
"""

# def assert_types(obj, name, expectedType):
#     """ Helper to assert proper typing of function inputs """
#     assert isinstance(obj, expectedType), f'{name} expected type {expectedType}, but found type {type{obj}}'

class GAN(object):

    def __init__(self, name, rowNum, columnNum, channelNum):
        self.name = name
        self.rowNum = rowNum
        self.columnNum = columnNum
        self.channelNum = channelNum

        self.discriminator = None
        self.generator = None
