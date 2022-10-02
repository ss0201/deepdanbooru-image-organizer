"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""Tools to help with the TensorFlow 2.0 transition.

This module is meant for TensorFlow internal implementation, not for users of
the TensorFlow library. For that see tf.compat instead.
"""
def enable(): # -> None:
  ...

def disable(): # -> None:
  ...

@tf_export("__internal__.tf2.enabled", v1=[])
def enabled():
  ...
