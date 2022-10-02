"""
This type stub file was generated by pyright.
"""

"""Helper classes that list&validate all attributes to serialize to SavedModel.
"""
base_layer = ...
training_lib = ...
metrics = ...
recurrent = ...
class SerializedAttributes:
  """Class that tracks and validates all serialization attributes.

  Keras models contain many Python-defined components. For example, the
  trainable_variable property lists the model's trainable variables by
  recursively retrieving the trainable variables from each of the child layers.
  Another example is model.call, a python function that calls child layers and
  adds ops to the backend graph.

  Only Tensorflow checkpointable objects and functions can be serialized to
  SavedModel. Serializing a Keras model as-is results in a checkpointable object
  that does not resemble a Keras model at all. Thus, extra checkpointable
  objects and functions must be created during serialization.

  **Defining new serialized attributes**
  Child classes should be defined using:
    SerializedAttributes.with_attributes(
        'name', checkpointable_objects=[...], functions=[...], copy_from=[...])
  This class is used to cache generated checkpointable objects and functions,
  ensuring that new objects and functions are generated a single time.

  **Usage during serialization**
  Each Layer/Model object should have a corresponding instance of
  SerializedAttributes. Create a new instance by calling
  `SerializedAttributes.new(obj)`. Objects and functions may be saved using
  `.set_and_validate_checkpointable_objects`/`.set_and_and_validate_functions`.
  The properties `.checkpointable_objects` and `.functions` returns the cached
  values.

  **Adding/changing attributes to save to SavedModel**
  1. Change the call to `SerializedAttributes.with_attributes` in the correct
     class:
     - CommonEndpoints: Base attributes to be added during serialization. If
       these attributes are present in a Trackable object, it can be
       deserialized to a Keras Model.
     - LayerAttributes: Attributes to serialize for Layer objects.
     - ModelAttributes: Attributes to serialize for Model objects.
  2. Update class docstring
  3. Update arguments to any calls to `set_and_validate_*`. For example, if
     `call_raw_tensors` is added to the ModelAttributes function list, then
     a `call_raw_tensors` function should be passed to
     `set_and_validate_functions`.

  **Common endpoints vs other attributes**
  Only common endpoints are attached directly to the root object. Keras-specific
  attributes are saved to a separate trackable object with the name "keras_api".
  The number of objects attached to the root is limited because any naming
  conflicts will cause user code to break.

  Another reason is that this will only affect users who call
  `tf.saved_model.load` instead of `tf.keras.models.load_model`. These are
  advanced users who are likely to have defined their own tf.functions and
  trackable objects. The added Keras-specific attributes are kept out of the way
  in the "keras_api" namespace.

  Properties defined in this class may be used to filter out keras-specific
  attributes:
  - `functions_to_serialize`: Returns dict of functions to attach to the root
      object.
  - `checkpointable_objects_to_serialize`: Returns dict of objects to attach to
      the root object (including separate trackable object containing
      keras-specific attributes)

  All changes to the serialized attributes must be backwards-compatible, so
  attributes should not be removed or modified without sufficient justification.
  """
  @staticmethod
  def with_attributes(name, checkpointable_objects=..., functions=..., copy_from=...): # -> Any:
    """Creates a subclass with all attributes as specified in the arguments.

    Args:
      name: Name of subclass
      checkpointable_objects: List of checkpointable objects to be serialized
        in the SavedModel.
      functions: List of functions to be serialized in the SavedModel.
      copy_from: List of other SerializedAttributes subclasses. The returned
        class will copy checkpoint objects/functions from each subclass.

    Returns:
      Child class with attributes as defined in the `checkpointable_objects`
      and `functions` lists.
    """
    ...
  
  @staticmethod
  def new(obj): # -> ModelAttributes | MetricAttributes | RNNAttributes | LayerAttributes:
    """Returns a new SerializedAttribute object."""
    ...
  
  def __init__(self) -> None:
    ...
  
  @property
  def functions(self): # -> dict[Unknown, Unknown]:
    """Returns dictionary of all functions."""
    ...
  
  @property
  def checkpointable_objects(self): # -> dict[Unknown, Unknown]:
    """Returns dictionary of all checkpointable objects."""
    ...
  
  @property
  def functions_to_serialize(self): # -> dict[Unknown, Unknown]:
    """Returns functions to attach to the root object during serialization."""
    ...
  
  @property
  def objects_to_serialize(self): # -> dict[Unknown, Unknown]:
    """Returns objects to attach to the root object during serialization."""
    ...
  
  def set_and_validate_functions(self, function_dict): # -> dict[Unknown, Unknown]:
    """Saves function dictionary, and validates dictionary values."""
    ...
  
  def set_and_validate_objects(self, object_dict): # -> dict[Unknown, Unknown]:
    """Saves objects to a dictionary, and validates the values."""
    ...
  


class CommonEndpoints(SerializedAttributes.with_attributes('CommonEndpoints', checkpointable_objects=['variables', 'trainable_variables', 'regularization_losses'], functions=['__call__', 'call_and_return_all_conditional_losses', '_default_save_signature'])):
  """Common endpoints shared by all models loadable by Keras.

  List of all attributes:
    variables: List of all variables in the model and its sublayers.
    trainable_variables: List of all trainable variables in the model and its
      sublayers.
    regularization_losses: List of all unconditional losses (losses not
      dependent on the inputs) in the model and its sublayers.
    __call__: Function that takes inputs and returns the outputs of the model
      call function.
    call_and_return_all_conditional_losses: Function that returns a tuple of
      (call function outputs, list of all losses that depend on the inputs).
    _default_save_signature: Traced model call function. This is only included
      if the top level exported object is a Keras model.
  """
  ...


class LayerAttributes(SerializedAttributes.with_attributes('LayerAttributes', checkpointable_objects=['non_trainable_variables', 'layers', 'metrics', 'layer_regularization_losses', 'layer_metrics'], functions=['call_and_return_conditional_losses', 'activity_regularizer_fn'], copy_from=[CommonEndpoints])):
  """Layer checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from CommonEndpoints
    non_trainable_variables: List of non-trainable variables in the layer and
      its sublayers.
    layers: List of all sublayers.
    metrics: List of all metrics in the layer and its sublayers.
    call_and_return_conditional_losses: Function that takes inputs and returns a
      tuple of (outputs of the call function, list of input-dependent losses).
      The list of losses excludes the activity regularizer function, which is
      separate to allow the deserialized Layer object to define a different
      activity regularizer.
    activity_regularizer_fn: Callable that returns the activity regularizer loss
    layer_regularization_losses: List of losses owned only by this layer.
    layer_metrics: List of metrics owned by this layer.
  """
  ...


class ModelAttributes(SerializedAttributes.with_attributes('ModelAttributes', copy_from=[LayerAttributes])):
  """Model checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from LayerAttributes (including CommonEndpoints)
  """
  ...


class MetricAttributes(SerializedAttributes.with_attributes('MetricAttributes', checkpointable_objects=['variables'], functions=[])):
  """Attributes that are added to Metric objects when saved to SavedModel.

  List of all attributes:
    variables: list of all variables
  """
  ...


class RNNAttributes(SerializedAttributes.with_attributes('RNNAttributes', checkpointable_objects=['states'], copy_from=[LayerAttributes])):
  """RNN checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from LayerAttributes (including CommonEndpoints)
    states: List of state variables
  """
  ...


