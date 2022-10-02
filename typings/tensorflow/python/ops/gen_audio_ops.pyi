"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: audio_ops.cc
"""
def audio_spectrogram(input, window_size, stride, magnitude_squared=..., name=...):
  r"""Produces a visualization of audio data over time.

  Spectrograms are a standard way of representing audio information as a series of
  slices of frequency information, one slice for each window of time. By joining
  these together into a sequence, they form a distinctive fingerprint of the sound
  over time.

  This op expects to receive audio data as an input, stored as floats in the range
  -1 to 1, together with a window width in samples, and a stride specifying how
  far to move the window between slices. From this it generates a three
  dimensional output. The first dimension is for the channels in the input, so a
  stereo audio input would have two here for example. The second dimension is time,
  with successive frequency slices. The third dimension has an amplitude value for
  each frequency during that time slice.

  This means the layout when converted and saved as an image is rotated 90 degrees
  clockwise from a typical spectrogram. Time is descending down the Y axis, and
  the frequency decreases from left to right.

  Each value in the result represents the square root of the sum of the real and
  imaginary parts of an FFT on the current window of samples. In this way, the
  lowest dimension represents the power of each frequency in the current window,
  and adjacent windows are concatenated in the next dimension.

  To get a more intuitive and visual look at what this operation does, you can run
  tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
  resulting spectrogram as a PNG image.

  Args:
    input: A `Tensor` of type `float32`. Float representation of audio data.
    window_size: An `int`.
      How wide the input window is in samples. For the highest efficiency
      this should be a power of two, but other values are accepted.
    stride: An `int`.
      How widely apart the center of adjacent sample windows should be.
    magnitude_squared: An optional `bool`. Defaults to `False`.
      Whether to return the squared magnitude or just the
      magnitude. Using squared magnitude can avoid extra calculations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  ...

AudioSpectrogram = ...
def audio_spectrogram_eager_fallback(input, window_size, stride, magnitude_squared, name, ctx):
  ...

_DecodeWavOutput = ...
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('audio.decode_wav')
def decode_wav(contents, desired_channels=..., desired_samples=..., name=...):
  r"""Decode a 16-bit PCM WAV file to a float tensor.

  The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.

  When desired_channels is set, if the input contains fewer channels than this
  then the last channel will be duplicated to give the requested number, else if
  the input has more channels than requested then the additional channels will be
  ignored.

  If desired_samples is set, then the audio will be cropped or padded with zeroes
  to the requested length.

  The first output contains a Tensor with the content of the audio samples. The
  lowest dimension will be the number of channels, and the second will be the
  number of samples. For example, a ten-sample-long stereo WAV file should give an
  output shape of [10, 2].

  Args:
    contents: A `Tensor` of type `string`.
      The WAV-encoded audio, usually from a file.
    desired_channels: An optional `int`. Defaults to `-1`.
      Number of sample channels wanted.
    desired_samples: An optional `int`. Defaults to `-1`.
      Length of audio requested.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (audio, sample_rate).

    audio: A `Tensor` of type `float32`.
    sample_rate: A `Tensor` of type `int32`.
  """
  ...

DecodeWav = ...
_dispatcher_for_decode_wav = decode_wav._tf_type_based_dispatcher.Dispatch
def decode_wav_eager_fallback(contents, desired_channels, desired_samples, name, ctx): # -> DecodeWav:
  ...

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('audio.encode_wav')
def encode_wav(audio, sample_rate, name=...): # -> _dispatcher_for_encode_wav | object:
  r"""Encode audio data using the WAV file format.

  This operation will generate a string suitable to be saved out to create a .wav
  audio file. It will be encoded in the 16-bit PCM format. It takes in float
  values in the range -1.0f to 1.0f, and any outside that value will be clamped to
  that range.

  `audio` is a 2-D float Tensor of shape `[length, channels]`.
  `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

  Args:
    audio: A `Tensor` of type `float32`. 2-D with shape `[length, channels]`.
    sample_rate: A `Tensor` of type `int32`.
      Scalar containing the sample frequency.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

EncodeWav = ...
_dispatcher_for_encode_wav = encode_wav._tf_type_based_dispatcher.Dispatch
def encode_wav_eager_fallback(audio, sample_rate, name, ctx):
  ...

def mfcc(spectrogram, sample_rate, upper_frequency_limit=..., lower_frequency_limit=..., filterbank_channel_count=..., dct_coefficient_count=..., name=...):
  r"""Transforms a spectrogram into a form that's useful for speech recognition.

  Mel Frequency Cepstral Coefficients are a way of representing audio data that's
  been effective as an input feature for machine learning. They are created by
  taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
  higher frequencies that are less significant to the human ear. They have a long
  history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  is a good resource to learn more.

  Args:
    spectrogram: A `Tensor` of type `float32`.
      Typically produced by the Spectrogram op, with magnitude_squared
      set to true.
    sample_rate: A `Tensor` of type `int32`.
      How many samples per second the source audio used.
    upper_frequency_limit: An optional `float`. Defaults to `4000`.
      The highest frequency to use when calculating the
      ceptstrum.
    lower_frequency_limit: An optional `float`. Defaults to `20`.
      The lowest frequency to use when calculating the
      ceptstrum.
    filterbank_channel_count: An optional `int`. Defaults to `40`.
      Resolution of the Mel bank used internally.
    dct_coefficient_count: An optional `int`. Defaults to `13`.
      How many output channels to produce per time slice.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  ...

Mfcc = ...
def mfcc_eager_fallback(spectrogram, sample_rate, upper_frequency_limit, lower_frequency_limit, filterbank_channel_count, dct_coefficient_count, name, ctx):
  ...

