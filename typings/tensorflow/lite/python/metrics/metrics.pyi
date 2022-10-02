"""
This type stub file was generated by pyright.
"""

import os
from typing import Optional, Text
from tensorflow.lite.python.metrics import metrics_interface
from tflite_runtime import metrics_interface

"""Python TFLite metrics helper."""
if notos.path.splitext(__file__)[0].endswith(os.path.join('tflite_runtime', 'metrics_portable')):
  ...
else:
  ...
class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  """TFLite metrics helper."""
  def __init__(self, model_hash: Optional[Text] = ..., model_path: Optional[Text] = ...) -> None:
    ...
  
  def increase_counter_debugger_creation(self): # -> None:
    ...
  
  def increase_counter_interpreter_creation(self): # -> None:
    ...
  
  def increase_counter_converter_attempt(self): # -> None:
    ...
  
  def increase_counter_converter_success(self): # -> None:
    ...
  
  def set_converter_param(self, name, value): # -> None:
    ...
  
  def set_converter_error(self, error_data): # -> None:
    ...
  
  def set_converter_latency(self, value): # -> None:
    ...
  


class TFLiteConverterMetrics(TFLiteMetrics):
  """Similar to TFLiteMetrics but specialized for converter."""
  def __del__(self): # -> None:
    ...
  
  def set_export_required(self): # -> None:
    ...
  
  def export_metrics(self): # -> None:
    ...
  


