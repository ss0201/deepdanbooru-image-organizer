"""
This type stub file was generated by pyright.
"""

import sys as _sys
from keras.legacy_tf_layers.migration_utils import DeterministicRandomTestTool
from keras.legacy_tf_layers.variable_scope_shim import get_or_create_layer, track_tf1_style_variables
from keras.utils.data_utils import GeneratorEnqueuer, OrderedEnqueuer, Sequence, SequenceEnqueuer, get_file, pad_sequences
from keras.utils.generic_utils import CustomObjectScope, Progbar, deserialize_keras_object, get_custom_objects, get_registered_name, get_registered_object, register_keras_serializable, serialize_keras_object
from keras.utils.image_utils import array_to_img, img_to_array, load_img, save_img
from keras.utils.io_utils import disable_interactive_logging, enable_interactive_logging, is_interactive_logging_enabled
from keras.utils.layer_utils import get_source_inputs
from keras.utils.np_utils import normalize, to_categorical
from keras.utils.vis_utils import model_to_dot, plot_model
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public Keras utilities.
"""
if notisinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...