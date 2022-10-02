"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export

"""Keras backend config API."""
_FLOATX = ...
_EPSILON = ...
_IMAGE_DATA_FORMAT = ...
@keras_export('keras.backend.epsilon')
@dispatch.add_dispatch_support
def epsilon(): # -> float:
  """Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  """
  ...

@keras_export('keras.backend.set_epsilon')
def set_epsilon(value): # -> None:
  """Sets the value of the fuzz factor used in numeric expressions.

  Args:
      value: float. New value of epsilon.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  >>> tf.keras.backend.set_epsilon(1e-5)
  >>> tf.keras.backend.epsilon()
  1e-05
   >>> tf.keras.backend.set_epsilon(1e-7)
  """
  ...

@keras_export('keras.backend.floatx')
def floatx(): # -> Literal['float32']:
  """Returns the default float type, as a string.

  E.g. `'float16'`, `'float32'`, `'float64'`.

  Returns:
      String, the current default float type.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  """
  ...

@keras_export('keras.backend.set_floatx')
def set_floatx(value): # -> None:
  """Sets the default float type.

  Note: It is not recommended to set this to float16 for training, as this will
  likely cause numeric stability issues. Instead, mixed precision, which is
  using a mix of float16 and float32, can be used by calling
  `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. See the
  [mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for details.

  Args:
      value: String; `'float16'`, `'float32'`, or `'float64'`.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  >>> tf.keras.backend.set_floatx('float64')
  >>> tf.keras.backend.floatx()
  'float64'
  >>> tf.keras.backend.set_floatx('float32')

  Raises:
      ValueError: In case of invalid value.
  """
  ...

@keras_export('keras.backend.image_data_format')
@dispatch.add_dispatch_support
def image_data_format(): # -> Literal['channels_last']:
  """Returns the default image data format convention.

  Returns:
      A string, either `'channels_first'` or `'channels_last'`

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  """
  ...

@keras_export('keras.backend.set_image_data_format')
def set_image_data_format(data_format): # -> None:
  """Sets the value of the image data format convention.

  Args:
      data_format: string. `'channels_first'` or `'channels_last'`.

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  >>> tf.keras.backend.set_image_data_format('channels_first')
  >>> tf.keras.backend.image_data_format()
  'channels_first'
  >>> tf.keras.backend.set_image_data_format('channels_last')

  Raises:
      ValueError: In case of invalid `data_format` value.
  """
  ...

