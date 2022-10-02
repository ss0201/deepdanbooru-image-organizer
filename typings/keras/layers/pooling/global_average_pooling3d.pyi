"""
This type stub file was generated by pyright.
"""

from keras.layers.pooling.base_global_pooling3d import GlobalPooling3D
from tensorflow.python.util.tf_export import keras_export

"""Global average pooling 3D layer."""
@keras_export("keras.layers.GlobalAveragePooling3D", "keras.layers.GlobalAvgPool3D")
class GlobalAveragePooling3D(GlobalPooling3D):
    """Global Average pooling operation for 3D data.

    Args:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      keepdims: A boolean, whether to keep the spatial dimensions or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the spatial dimensions are retained with
        length 1.
        The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, channels)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          5D tensor with shape `(batch_size, 1, 1, 1, channels)`
        - If `data_format='channels_first'`:
          5D tensor with shape `(batch_size, channels, 1, 1, 1)`
    """
    def call(self, inputs):
        ...
    


GlobalAvgPool3D = GlobalAveragePooling3D
