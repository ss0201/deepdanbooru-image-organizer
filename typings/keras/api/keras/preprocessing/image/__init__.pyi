"""
This type stub file was generated by pyright.
"""

import sys as _sys
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator, Iterator, NumpyArrayIterator, apply_affine_transform, apply_brightness_shift, apply_channel_shift, random_brightness, random_channel_shift, random_rotation, random_shear, random_shift, random_zoom
from keras.utils.image_utils import array_to_img, img_to_array, load_img, save_img
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Utilies for image preprocessing and augmentation.

Deprecated: `tf.keras.preprocessing.image` APIs do not operate on tensors and
are not recommended for new code. Prefer loading data with
`tf.keras.utils.image_dataset_from_directory`, and then transforming the output
`tf.data.Dataset` with preprocessing layers. For more information, see the
tutorials for [loading images](
https://www.tensorflow.org/tutorials/load_data/images) and [augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as the
[preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

"""
if notisinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...