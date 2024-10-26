import tensorflow as tf
from keras.layers import Layer
import keras.backend as K

class spp(Layer):
    def __init__(self, pool_list, **kwargs):
        """
        Spatial Pyramid Pooling Layer
        Args:
            pool_list: List of pooling levels to use, e.g., [1, 2, 4]
        """
        self.pool_list = pool_list
        super(SPP, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SPP, self).build(input_shape)

    def call(self, x):
        """
        Apply the spatial pyramid pooling on the input tensor.
        """
        pool_outputs = []
        shape = K.int_shape(x)
        h, w = shape[1], shape[2]

        for pool_size in self.pool_list:
            kh = h // pool_size
            kw = w // pool_size

            # Define the pooling layer with `pool_size` and `strides`
            pool = tf.nn.max_pool2d(x, ksize=[1, kh, kw, 1], strides=[1, kh, kw, 1], padding='SAME')
            pool_outputs.append(K.flatten(pool))

        # Concatenate all pooled outputs
        return K.concatenate(pool_outputs)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the SPP layer.
        """
        num_features = input_shape[-1] * sum([i * i for i in self.pool_list])
        return (input_shape[0], num_features)

# To use this in your model file:
# from spatial_pyramid_pooling import SPP
# and replace `spp` in your code with `SPP([1, 2, 4])` or the pooling levels you prefer.
