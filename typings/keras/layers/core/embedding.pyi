"""
This type stub file was generated by pyright.
"""

from keras.dtensor import utils
from keras.engine.base_layer import Layer
from keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export

"""Embedding layer."""
@keras_export("keras.layers.Embedding")
class Embedding(Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used on positive integer inputs of a fixed range. The
    `tf.keras.layers.TextVectorization`, `tf.keras.layers.StringLookup`,
    and `tf.keras.layers.IntegerLookup` preprocessing layers can help prepare
    inputs for an `Embedding` layer.

    This layer accepts `tf.Tensor` and `tf.RaggedTensor` inputs. It cannot be
    called with `tf.SparseTensor` input.

    Example:

    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
    >>> # The model will take as input an integer matrix of size (batch,
    >>> # input_length), and the largest integer (i.e. word index) in the input
    >>> # should be no larger than 999 (vocabulary size).
    >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
    >>> # dimension.
    >>> input_array = np.random.randint(1000, size=(32, 10))
    >>> model.compile('rmsprop', 'mse')
    >>> output_array = model.predict(input_array)
    >>> print(output_array.shape)
    (32, 10, 64)

    Args:
      input_dim: Integer. Size of the vocabulary,
        i.e. maximum integer index + 1.
      output_dim: Integer. Dimension of the dense embedding.
      embeddings_initializer: Initializer for the `embeddings`
        matrix (see `keras.initializers`).
      embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix (see `keras.regularizers`).
      embeddings_constraint: Constraint function applied to
        the `embeddings` matrix (see `keras.constraints`).
      mask_zero: Boolean, whether or not the input value 0 is a special
        "padding" value that should be masked out. This is useful when using
        recurrent layers which may take variable length input. If this is
        `True`, then all subsequent layers in the model need to support masking
        or an exception will be raised. If mask_zero is set to True, as a
        consequence, index 0 cannot be used in the vocabulary (input_dim should
        equal size of vocabulary + 1).
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).

    Input shape:
      2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
      3D tensor with shape: `(batch_size, input_length, output_dim)`.

    **Note on variable placement:**
    By default, if a GPU is available, the embedding matrix will be placed on
    the GPU. This achieves the best performance, but it might cause issues:

    - You may be using an optimizer that does not support sparse GPU kernels.
    In this case you will see an error upon training your model.
    - Your embedding matrix may be too large to fit on your GPU. In this case
    you will see an Out Of Memory (OOM) error.

    In such cases, you should place the embedding matrix on the CPU memory.
    You can do so with a device scope, as such:

    ```python
    with tf.device('cpu:0'):
      embedding_layer = Embedding(...)
      embedding_layer.build()
    ```

    The pre-built `embedding_layer` instance can then be added to a `Sequential`
    model (e.g. `model.add(embedding_layer)`), called in a Functional model
    (e.g. `x = embedding_layer(x)`), or used in a subclassed model.
    """
    @utils.allow_initializer_layout
    def __init__(self, input_dim, output_dim, embeddings_initializer=..., embeddings_regularizer=..., activity_regularizer=..., embeddings_constraint=..., mask_zero=..., input_length=..., **kwargs) -> None:
        ...
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape=...): # -> None:
        ...
    
    def compute_mask(self, inputs, mask=...): # -> None:
        ...
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape): # -> tuple[Unknown, ...]:
        ...
    
    def call(self, inputs):
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    


