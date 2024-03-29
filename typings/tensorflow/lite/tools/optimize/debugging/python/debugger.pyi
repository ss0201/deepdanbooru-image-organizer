"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, Callable, IO, Iterable, List, Mapping, Optional, Sequence
from tensorflow.python.util import tf_export

"""Python TF-Lite QuantizationDebugger."""
TFLiteConverter = Any
_DEFAULT_LAYER_DEBUG_METRICS = ...
_NUMERIC_VERIFY_OP_NAME = ...
@tf_export.tf_export('lite.experimental.QuantizationDebugOptions')
class QuantizationDebugOptions:
  """Debug options to set up a given QuantizationDebugger."""
  def __init__(self, layer_debug_metrics: Optional[Mapping[str, Callable[[np.ndarray], float]]] = ..., model_debug_metrics: Optional[Mapping[str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], float]]] = ..., layer_direct_compare_metrics: Optional[Mapping[str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray], float, int], float]]] = ..., denylisted_ops: Optional[List[str]] = ..., denylisted_nodes: Optional[List[str]] = ..., fully_quantize: bool = ...) -> None:
    """Initializes debugger options.

    Args:
      layer_debug_metrics: a dict to specify layer debug functions
        {function_name_str: function} where the function accepts result of
          NumericVerify Op, which is value difference between float and
          dequantized op results. The function returns single scalar value.
      model_debug_metrics: a dict to specify model debug functions
        {function_name_str: function} where the function accepts outputs from
          two models, and returns single scalar value for a metric. (e.g.
          accuracy, IoU)
      layer_direct_compare_metrics: a dict to specify layer debug functions
        {function_name_str: function}. The signature is different from that of
          `layer_debug_metrics`, and this one gets passed (original float value,
          original quantized value, scale, zero point). The function's
          implementation is responsible for correctly dequantize the quantized
          value to compare. Use this one when comparing diff is not enough.
          (Note) quantized value is passed as int8, so cast to int32 is needed.
      denylisted_ops: a list of op names which is expected to be removed from
        quantization.
      denylisted_nodes: a list of op's output tensor names to be removed from
        quantization.
      fully_quantize: Bool indicating whether to fully quantize the model.
        Besides model body, the input/output will be quantized as well.
        Corresponding to mlir_quantize's fully_quantize parameter.

    Raises:
      ValueError: when there are duplicate keys
    """
    ...
  


@tf_export.tf_export('lite.experimental.QuantizationDebugger')
class QuantizationDebugger:
  """Debugger for Quantized TensorFlow Lite debug mode models.

  This can run the TensorFlow Lite converted models equipped with debug ops and
  collect debug information. This debugger calculates statistics from
  user-defined post-processing functions as well as default ones.
  """
  def __init__(self, quant_debug_model_path: Optional[str] = ..., quant_debug_model_content: Optional[bytes] = ..., float_model_path: Optional[str] = ..., float_model_content: Optional[bytes] = ..., debug_dataset: Optional[Callable[[], Iterable[Sequence[np.ndarray]]]] = ..., debug_options: Optional[QuantizationDebugOptions] = ..., converter: Optional[TFLiteConverter] = ...) -> None:
    """Runs the TFLite debugging model with given debug options.

    Args:
      quant_debug_model_path: Path to the quantized debug TFLite model file.
      quant_debug_model_content: Content of the quantized debug TFLite model.
      float_model_path: Path to float TFLite model file.
      float_model_content: Content of the float TFLite model.
      debug_dataset: a factory function that returns dataset generator which is
        used to generate input samples (list of np.ndarray) for the model. The
        generated elements must have same types and shape as inputs to the
        model.
      debug_options: Debug options to debug the given model.
      converter: Optional, use converter instead of quantized model.

    Raises:
      ValueError: If the debugger was unable to be created.

    Attributes:
      layer_statistics: results of error metrics for each NumericVerify op
        results. in {layer_name: {metric_name: metric}} format.
      model_statistics: results of error metrics for difference between float
        and quantized models. in {metric_name: metric} format.
    """
    ...
  
  @property
  def options(self) -> QuantizationDebugOptions:
    ...
  
  @options.setter
  def options(self, options: QuantizationDebugOptions) -> None:
    ...
  
  def get_nondebug_quantized_model(self) -> bytes:
    """Returns a non-instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for nondebug model. The model will not be instrumented with
    numeric verification operations.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    ...
  
  def get_debug_quantized_model(self) -> bytes:
    """Returns an instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for model. The model will be instrumented with numeric
    verification operations and should only be used for debugging.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    ...
  
  def run(self) -> None:
    """Runs models and gets metrics."""
    ...
  
  def layer_statistics_dump(self, file: IO[str]) -> None:
    """Dumps layer statistics into file, in csv format.

    Args:
      file: file, or file-like object to write.
    """
    ...
  


