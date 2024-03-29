"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import keras_export

"""EfficientNet V2 models for Keras.

Reference:
- [EfficientNetV2: Smaller Models and Faster Training](
    https://arxiv.org/abs/2104.00298) (ICML 2021)
"""
BASE_WEIGHTS_PATH = ...
WEIGHTS_HASHES = ...
DEFAULT_BLOCKS_ARGS = ...
CONV_KERNEL_INITIALIZER = ...
DENSE_KERNEL_INITIALIZER = ...
BASE_DOCSTRING = ...
def round_filters(filters, width_coefficient, min_depth, depth_divisor): # -> int:
    """Round number of filters based on depth multiplier."""
    ...

def round_repeats(repeats, depth_coefficient): # -> int:
    """Round number of repeats based on depth multiplier."""
    ...

def MBConvBlock(input_filters: int, output_filters: int, expand_ratio=..., kernel_size=..., strides=..., se_ratio=..., bn_momentum=..., activation=..., survival_probability: float = ..., name=...): # -> (inputs: Unknown) -> (Unknown | None):
    """MBConv block: Mobile Inverted Residual Bottleneck."""
    ...

def FusedMBConvBlock(input_filters: int, output_filters: int, expand_ratio=..., kernel_size=..., strides=..., se_ratio=..., bn_momentum=..., activation=..., survival_probability: float = ..., name=...): # -> (inputs: Unknown) -> (Unknown | None):
    """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a
    conv2d."""
    ...

def EfficientNetV2(width_coefficient, depth_coefficient, default_size, dropout_rate=..., drop_connect_rate=..., depth_divisor=..., min_depth=..., bn_momentum=..., activation=..., blocks_args=..., model_name=..., include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    """Instantiates the EfficientNetV2 architecture using given scaling
    coefficients.

    Args:
      width_coefficient: float, scaling coefficient for network width.
      depth_coefficient: float, scaling coefficient for network depth.
      default_size: integer, default input image size.
      dropout_rate: float, dropout rate before final classifier layer.
      drop_connect_rate: float, dropout rate at skip connections.
      depth_divisor: integer, a unit of network width.
      min_depth: integer, minimum number of filters.
      bn_momentum: float. Momentum parameter for Batch Normalization layers.
      activation: activation function.
      blocks_args: list of dicts, parameters to construct block modules.
      model_name: string, model name.
      include_top: whether to include the fully-connected layer at the top of
        the network.
      weights: one of `None` (random initialization), `"imagenet"` (pre-training
        on ImageNet), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) or
        numpy array to use as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top`
        is False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor output
          of the last convolutional layer.
        - "avg" means that global average pooling will be applied to the output
          of the last convolutional layer, and thus the output of the model will
          be a 2D tensor.
        - `"max"` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
      classifier_activation: A string or callable. The activation function to
        use on the `"top"` layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the `"top"` layer.
      include_preprocessing: Boolean, whether to include the preprocessing layer
        (`Rescaling`) at the bottom of the network. Defaults to `True`.

    Returns:
      A `keras.Model` instance.

    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `"softmax"` or `None` when
        using a pretrained top layer.
    """
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2B0", "keras.applications.EfficientNetV2B0")
def EfficientNetV2B0(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2B1", "keras.applications.EfficientNetV2B1")
def EfficientNetV2B1(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2B2", "keras.applications.EfficientNetV2B2")
def EfficientNetV2B2(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2B3", "keras.applications.EfficientNetV2B3")
def EfficientNetV2B3(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2S", "keras.applications.EfficientNetV2S")
def EfficientNetV2S(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2M", "keras.applications.EfficientNetV2M")
def EfficientNetV2M(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.EfficientNetV2L", "keras.applications.EfficientNetV2L")
def EfficientNetV2L(include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=..., include_preprocessing=...):
    ...

@keras_export("keras.applications.efficientnet_v2.preprocess_input")
def preprocess_input(x, data_format=...):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the EfficientNetV2 model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.

    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. Defaults to
        None, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to "channels_last").{mode}

    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    """
    ...

@keras_export("keras.applications.efficientnet_v2.decode_predictions")
def decode_predictions(preds, top=...): # -> list[Unknown]:
    ...

