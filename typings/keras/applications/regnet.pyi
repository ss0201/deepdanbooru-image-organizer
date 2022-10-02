"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import keras_export

"""RegNet models for Keras.

References:

- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
  (CVPR 2020)
- [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
  (CVPR 2021)
"""
BASE_WEIGHTS_PATH = ...
WEIGHTS_HASHES = ...
MODEL_CONFIGS = ...
BASE_DOCSTRING = ...
def PreStem(name=...): # -> (x: Unknown) -> (Unknown | None):
    """Rescales and normalizes inputs to [0,1] and ImageNet mean and std.

    Args:
      name: name prefix

    Returns:
      Rescaled and normalized tensor
    """
    ...

def Stem(name=...): # -> (x: Unknown) -> (Unknown | None):
    """Implementation of RegNet stem.

    (Common to all model variants)
    Args:
      name: name prefix

    Returns:
      Output tensor of the Stem
    """
    ...

def SqueezeAndExciteBlock(filters_in, se_filters, name=...): # -> (inputs: Unknown) -> Unknown:
    """Implements the Squeeze and excite block (https://arxiv.org/abs/1709.01507).

    Args:
      filters_in: input filters to the block
      se_filters: filters to squeeze to
      name: name prefix

    Returns:
      A function object
    """
    ...

def XBlock(filters_in, filters_out, group_width, stride=..., name=...): # -> (inputs: Unknown) -> (Unknown | None):
    """Implementation of X Block.

    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      name: name prefix
    Returns:
      Output tensor of the block
    """
    ...

def YBlock(filters_in, filters_out, group_width, stride=..., squeeze_excite_ratio=..., name=...): # -> (inputs: Unknown) -> (Unknown | None):
    """Implementation of Y Block.

    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      name: name prefix
    Returns:
      Output tensor of the block
    """
    ...

def ZBlock(filters_in, filters_out, group_width, stride=..., squeeze_excite_ratio=..., bottleneck_ratio=..., name=...): # -> (inputs: Unknown) -> (Unknown | None):
    """Implementation of Z block Reference: [Fast and Accurate Model
    Scaling](https://arxiv.org/abs/2103.06877).

    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      bottleneck_ratio: inverted bottleneck ratio
      name: name prefix
    Returns:
      Output tensor of the block
    """
    ...

def Stage(block_type, depth, group_width, filters_in, filters_out, name=...): # -> (inputs: Unknown) -> (Unknown | None):
    """Implementation of Stage in RegNet.

    Args:
      block_type: must be one of "X", "Y", "Z"
      depth: depth of stage, number of blocks to use
      group_width: group width of all blocks in  this stage
      filters_in: input filters to this stage
      filters_out: output filters from this stage
      name: name prefix

    Returns:
      Output tensor of Stage
    """
    ...

def Head(num_classes=..., name=...): # -> (x: Unknown) -> (Unknown | None):
    """Implementation of classification head of RegNet.

    Args:
      num_classes: number of classes for Dense layer
      name: name prefix

    Returns:
      Output logits tensor.
    """
    ...

def RegNet(depths, widths, group_width, block_type, default_size, model_name=..., include_preprocessing=..., include_top=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    """Instantiates RegNet architecture given specific configuration.

    Args:
      depths: An iterable containing depths for each individual stages.
      widths: An iterable containing output channel width of each individual
        stages
      group_width: Number of channels to be used in each group. See grouped
        convolutions for more information.
      block_type: Must be one of `{"X", "Y", "Z"}`. For more details see the
        papers "Designing network design spaces" and "Fast and Accurate Model
        Scaling"
      default_size: Default input image size.
      model_name: An optional name for the model.
      include_preprocessing: boolean denoting whther to include preprocessing in
        the model
      include_top: Boolean denoting whether to include classification head to
        the model.
      weights: one of `None` (random initialization), "imagenet" (pre-training
        on ImageNet), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top`
        is False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`. - `None` means that the output of the model will be the 4D
        tensor output of the last convolutional layer. - `avg` means that global
        average pooling will be applied to the output of the last convolutional
        layer, and thus the output of the model will be a 2D tensor. - `max`
        means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
          using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000.
        ValueError: if `block_type` is not one of `{"X", "Y", "Z"}`

    """
    ...

@keras_export("keras.applications.regnet.RegNetX002", "keras.applications.RegNetX002")
def RegNetX002(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX004", "keras.applications.RegNetX004")
def RegNetX004(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX006", "keras.applications.RegNetX006")
def RegNetX006(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX008", "keras.applications.RegNetX008")
def RegNetX008(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX016", "keras.applications.RegNetX016")
def RegNetX016(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX032", "keras.applications.RegNetX032")
def RegNetX032(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX040", "keras.applications.RegNetX040")
def RegNetX040(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX064", "keras.applications.RegNetX064")
def RegNetX064(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX080", "keras.applications.RegNetX080")
def RegNetX080(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX120", "keras.applications.RegNetX120")
def RegNetX120(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX160", "keras.applications.RegNetX160")
def RegNetX160(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetX320", "keras.applications.RegNetX320")
def RegNetX320(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY002", "keras.applications.RegNetY002")
def RegNetY002(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY004", "keras.applications.RegNetY004")
def RegNetY004(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY006", "keras.applications.RegNetY006")
def RegNetY006(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY008", "keras.applications.RegNetY008")
def RegNetY008(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY016", "keras.applications.RegNetY016")
def RegNetY016(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY032", "keras.applications.RegNetY032")
def RegNetY032(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY040", "keras.applications.RegNetY040")
def RegNetY040(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY064", "keras.applications.RegNetY064")
def RegNetY064(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY080", "keras.applications.RegNetY080")
def RegNetY080(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY120", "keras.applications.RegNetY120")
def RegNetY120(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY160", "keras.applications.RegNetY160")
def RegNetY160(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.RegNetY320", "keras.applications.RegNetY320")
def RegNetY320(model_name=..., include_top=..., include_preprocessing=..., weights=..., input_tensor=..., input_shape=..., pooling=..., classes=..., classifier_activation=...): # -> Model:
    ...

@keras_export("keras.applications.regnet.preprocess_input")
def preprocess_input(x, data_format=...):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the regnet model
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

@keras_export("keras.applications.regnet.decode_predictions")
def decode_predictions(preds, top=...): # -> list[Unknown]:
    ...
