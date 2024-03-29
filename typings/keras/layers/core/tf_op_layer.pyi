"""
This type stub file was generated by pyright.
"""

import tensorflow.compat.v2 as tf
from keras.engine.base_layer import Layer

"""Contains the TFOpLambda layer."""
class ClassMethod(Layer):
    """Wraps a TF API Class's class method  in a `Layer` object.

    It is inserted by the Functional API construction whenever users call
    a supported TF Class's class method on KerasTensors.

    This is useful in the case where users do something like:
    x = keras.Input(...)
    y = keras.Input(...)
    out = tf.RaggedTensor.from_row_splits(x, y)
    """
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, cls_ref, method_name, **kwargs) -> None:
        ...
    
    def call(self, args, kwargs): # -> Any:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config, custom_objects=...): # -> Self@ClassMethod:
        ...
    


class KerasOpDispatcher(tf.__internal__.dispatch.GlobalOpDispatcher):
    """A global dispatcher that allows building a functional model with TF
    Ops."""
    def handle(self, op, args, kwargs): # -> None:
        """Handle the specified operation with the specified arguments."""
        ...
    


class InstanceProperty(Layer):
    """Wraps an instance property access (e.g.

    `x.foo`) in a Keras Layer.

    This layer takes an attribute name `attr_name` in the constructor and,
    when called on input tensor `obj` returns `obj.attr_name`.

    KerasTensors specialized for specific extension types use it to
    represent instance property accesses on the represented object in the
    case where the property needs to be dynamically accessed as opposed to
    being statically computed from the typespec, e.g.

    x = keras.Input(..., ragged=True)
    out = x.flat_values
    """
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, attr_name, **kwargs) -> None:
        ...
    
    def call(self, obj): # -> Any:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config, custom_objects=...): # -> Self@InstanceProperty:
        ...
    


class InstanceMethod(InstanceProperty):
    """Wraps an instance method access (e.g. `x.foo(arg)` in a Keras Layer.

    This layer takes an attribute name `attr_name` in the constructor and,
    when called on input tensor `obj` with additional arguments `args` and
    `kwargs` returns `obj.attr_name(*args, **kwargs)`.

    KerasTensors specialized for specific extension types use it to
    represent dynamic instance method calls on the represented object, e.g.

    x = keras.Input(..., ragged=True)
    new_values = keras.Input(...)
    out = x.with_values(new_values)
    """
    def call(self, obj, args, kwargs): # -> Any:
        ...
    


class TFOpLambda(Layer):
    """Wraps TF API symbols in a `Layer` object.

    It is inserted by the Functional API construction whenever users call
    a supported TF symbol on KerasTensors.

    Like Lambda layers, this layer tries to raise warnings when it detects users
    explicitly use variables in the call. (To let them know
    that the layer will not capture the variables).

    This is useful in the case where users do something like:
    x = keras.Input(...)
    y = tf.Variable(...)
    out = x * tf_variable
    """
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, function, **kwargs) -> None:
        ...
    
    def get_config(self): # -> dict[str, Unknown]:
        ...
    
    @classmethod
    def from_config(cls, config, custom_objects=...): # -> Self@TFOpLambda:
        ...
    


class TFClassMethodDispatcher(tf.__internal__.dispatch.OpDispatcher):
    """A class method dispatcher that allows building a functional model with TF
    class methods."""
    def __init__(self, cls, method_name) -> None:
        ...
    
    def handle(self, args, kwargs): # -> None:
        """Handle the specified operation with the specified arguments."""
        ...
    


class SlicingOpLambda(TFOpLambda):
    """Wraps TF API symbols in a `Layer` object.

    It is inserted by the Functional API construction whenever users call
    a supported TF symbol on KerasTensors.

    Like Lambda layers, this layer tries to raise warnings when it detects users
    explicitly use variables in the call. (To let them know
    that the layer will not capture the variables).

    This is useful in the case where users do something like:
    x = keras.Input(...)
    y = tf.Variable(...)
    out = x * tf_variable
    """
    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, function, **kwargs) -> None:
        ...
    


class TFSlicingOpDispatcher(tf.__internal__.dispatch.OpDispatcher):
    """A global dispatcher that allows building a functional model with TF
    Ops."""
    def __init__(self, op) -> None:
        ...
    
    def handle(self, args, kwargs): # -> None:
        """Handle the specified operation with the specified arguments."""
        ...
    


