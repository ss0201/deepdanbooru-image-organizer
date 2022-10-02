"""
This type stub file was generated by pyright.
"""

"""Keras SavedModel serialization."""
base_layer = ...
training_lib = ...
def save(model, filepath, overwrite, include_optimizer, signatures=..., options=..., save_traces=...): # -> None:
    """Saves a model as a SavedModel to the filepath.

    Args:
      model: Keras model instance to be saved.
      filepath: String path to save the model.
      overwrite: whether to overwrite the existing filepath.
      include_optimizer: If True, save the model's optimizer state.
      signatures: Signatures to save with the SavedModel. Applicable to the 'tf'
        format only. Please see the `signatures` argument in
        `tf.saved_model.save` for details.
      options: (only applies to SavedModel format) `tf.saved_model.SaveOptions`
        object that specifies options for saving to SavedModel.
      save_traces: (only applies to SavedModel format) When enabled, the
        SavedModel will store the function traces for each layer. This
        can be disabled, so that only the configs of each layer are stored.
        Defaults to `True`. Disabling this will decrease serialization time
        and reduce file size, but it requires that all custom layers/models
        implement a `get_config()` method.

    Raises:
      ValueError: if the model's inputs have not been defined.
    """
    ...

def generate_keras_metadata(saved_nodes, node_paths): # -> SavedMetadata:
    """Constructs a KerasMetadata proto with the metadata of each keras
    object."""
    ...

