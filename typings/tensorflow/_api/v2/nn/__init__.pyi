"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import experimental
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DeviceWrapper as RNNCellDeviceWrapper, DropoutWrapper as RNNCellDropoutWrapper, ResidualWrapper as RNNCellResidualWrapper
from tensorflow.python.ops.array_ops import depth_to_space_v2 as depth_to_space, space_to_batch_v2 as space_to_batch, space_to_depth_v2 as space_to_depth
from tensorflow.python.ops.candidate_sampling_ops import all_candidate_sampler, compute_accidental_hits, fixed_unigram_candidate_sampler, learned_unigram_candidate_sampler
from tensorflow.python.ops.ctc_ops import collapse_repeated, ctc_beam_search_decoder_v2 as ctc_beam_search_decoder, ctc_greedy_decoder, ctc_loss_v3 as ctc_loss, ctc_unique_labels
from tensorflow.python.ops.embedding_ops import embedding_lookup_sparse_v2 as embedding_lookup_sparse, embedding_lookup_v2 as embedding_lookup, safe_embedding_lookup_sparse_v2 as safe_embedding_lookup_sparse
from tensorflow.python.ops.gen_math_ops import tanh
from tensorflow.python.ops.gen_nn_ops import elu, l2_loss, lrn, relu, selu, softsign
from tensorflow.python.ops.math_ops import sigmoid, softplus
from tensorflow.python.ops.nn_impl import batch_norm_with_global_normalization_v2 as batch_norm_with_global_normalization, batch_normalization, compute_average_loss, depthwise_conv2d_v2 as depthwise_conv2d, l2_normalize, log_poisson_loss, moments_v2 as moments, nce_loss_v2 as nce_loss, normalize_moments, sampled_softmax_loss_v2 as sampled_softmax_loss, scale_regularization_loss, separable_conv2d_v2 as separable_conv2d, sigmoid_cross_entropy_with_logits_v2 as sigmoid_cross_entropy_with_logits, sufficient_statistics_v2 as sufficient_statistics, swish, weighted_cross_entropy_with_logits_v2 as weighted_cross_entropy_with_logits, weighted_moments_v2 as weighted_moments, zero_fraction
from tensorflow.python.ops.nn_ops import approx_max_k, approx_min_k, atrous_conv2d, atrous_conv2d_transpose, avg_pool1d, avg_pool2d, avg_pool3d, avg_pool_v2 as avg_pool, bias_add, conv1d_transpose, conv1d_v2 as conv1d, conv2d_transpose_v2 as conv2d_transpose, conv2d_v2 as conv2d, conv3d_transpose_v2 as conv3d_transpose, conv3d_v2 as conv3d, conv_transpose, convolution_v2 as convolution, crelu_v2 as crelu, depthwise_conv2d_native_backprop_filter as depthwise_conv2d_backprop_filter, depthwise_conv2d_native_backprop_input as depthwise_conv2d_backprop_input, dilation2d_v2 as dilation2d, dropout_v2 as dropout, erosion2d_v2 as erosion2d, fractional_avg_pool_v2 as fractional_avg_pool, fractional_max_pool_v2 as fractional_max_pool, gelu, in_top_k_v2 as in_top_k, isotonic_regression, leaky_relu, log_softmax_v2 as log_softmax, max_pool1d, max_pool2d, max_pool3d, max_pool_v2 as max_pool, max_pool_with_argmax_v2 as max_pool_with_argmax, pool_v2 as pool, relu6, softmax_cross_entropy_with_logits_v2 as softmax_cross_entropy_with_logits, softmax_v2 as softmax, sparse_softmax_cross_entropy_with_logits_v2 as sparse_softmax_cross_entropy_with_logits, top_k, with_space_to_batch

"""Primitive Neural Net (NN) Operations.

## Notes on padding

Several neural network operations, such as `tf.nn.conv2d` and
`tf.nn.max_pool2d`, take a `padding` parameter, which controls how the input is
padded before running the operation. The input is padded by inserting values
(typically zeros) before and after the tensor in each spatial dimension. The
`padding` parameter can either be the string `'VALID'`, which means use no
padding, or `'SAME'` which adds padding according to a formula which is
described below. Certain ops also allow the amount of padding per dimension to
be explicitly specified by passing a list to `padding`.

In the case of convolutions, the input is padded with zeros. In case of pools,
the padded input values are ignored. For example, in a max pool, the sliding
window ignores padded values, which is equivalent to the padded values being
`-infinity`.

### `'VALID'` padding

Passing `padding='VALID'` to an op causes no padding to be used. This causes the
output size to typically be smaller than the input size, even when the stride is
one. In the 2D case, the output size is computed as:

```python
out_height = ceil((in_height - filter_height + 1) / stride_height)
out_width  = ceil((in_width - filter_width + 1) / stride_width)
```

The 1D and 3D cases are similar. Note `filter_height` and `filter_width` refer
to the filter size after dilations (if any) for convolutions, and refer to the
window size for pools.

### `'SAME'` padding

With `'SAME'` padding, padding is applied to each spatial dimension. When the
strides are 1, the input is padded such that the output size is the same as the
input size. In the 2D case, the output size is computed as:

```python
out_height = ceil(in_height / stride_height)
out_width  = ceil(in_width / stride_width)
```

The amount of padding used is the smallest amount that results in the output
size. The formula for the total amount of padding per dimension is:

```python
if (in_height % strides[1] == 0):
  pad_along_height = max(filter_height - stride_height, 0)
else:
  pad_along_height = max(filter_height - (in_height % stride_height), 0)
if (in_width % strides[2] == 0):
  pad_along_width = max(filter_width - stride_width, 0)
else:
  pad_along_width = max(filter_width - (in_width % stride_width), 0)
```

Finally, the padding on the top, bottom, left and right are:

```python
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left
```

Note that the division by 2 means that there might be cases when the padding on
both sides (top vs bottom, right vs left) are off by one. In this case, the
bottom and right sides always get the one additional padded pixel. For example,
when pad_along_height is 5, we pad 2 pixels at the top and 3 pixels at the
bottom. Note that this is different from existing libraries such as PyTorch and
Caffe, which explicitly specify the number of padded pixels and always pad the
same number of pixels on both sides.

Here is an example of `'SAME'` padding:

>>> in_height = 5
>>> filter_height = 3
>>> stride_height = 2
>>>
>>> in_width = 2
>>> filter_width = 2
>>> stride_width = 1
>>>
>>> inp = tf.ones((2, in_height, in_width, 2))
>>> filter = tf.ones((filter_height, filter_width, 2, 2))
>>> strides = [stride_height, stride_width]
>>> output = tf.nn.conv2d(inp, filter, strides, padding='SAME')
>>> output.shape[1]  # output_height: ceil(5 / 2)
3
>>> output.shape[2] # output_width: ceil(2 / 1)
2

### Explicit padding

Certain ops, like `tf.nn.conv2d`, also allow a list of explicit padding amounts
to be passed to the `padding` parameter. This list is in the same format as what
is passed to `tf.pad`, except the padding must be a nested list, not a tensor.
For example, in the 2D case, the list is in the format `[[0, 0], [pad_top,
pad_bottom], [pad_left, pad_right], [0, 0]]` when `data_format` is its default
value of `'NHWC'`. The two `[0, 0]` pairs  indicate the batch and channel
dimensions have no padding, which is required, as only spatial dimensions can
have padding.

For example:

>>> inp = tf.ones((1, 3, 3, 1))
>>> filter = tf.ones((2, 2, 1, 1))
>>> strides = [1, 1]
>>> padding = [[0, 0], [1, 2], [0, 1], [0, 0]]
>>> output = tf.nn.conv2d(inp, filter, strides, padding=padding)
>>> tuple(output.shape)
(1, 5, 3, 1)
>>> # Equivalently, tf.pad can be used, since convolutions pad with zeros.
>>> inp = tf.pad(inp, padding)
>>> # 'VALID' means to use no padding in conv2d (we already padded inp)
>>> output2 = tf.nn.conv2d(inp, filter, strides, padding='VALID')
>>> tf.debugging.assert_equal(output, output2)

### Difference between convolution and pooling layers
How padding is used in convolution layers and pooling layers is different. For
convolution layers, padding is filled with values of zero, and padding is
multiplied with kernels. For pooling layers, padding is excluded from the
computation. For example when applying average pooling to a 4x4 grid, how much
padding is added will not impact the output. Here is an example that
demonstrates the difference.

>>> x_in = np.array([[
...   [[2], [2]],
...   [[1], [1]],
...   [[1], [1]]]])
>>> kernel_in = np.array([  # simulate the avg_pool with conv2d
...  [ [[0.25]], [[0.25]] ],
...  [ [[0.25]], [[0.25]] ]])
>>> x = tf.constant(x_in, dtype=tf.float32)
>>> kernel = tf.constant(kernel_in, dtype=tf.float32)
>>> conv_out = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
>>> pool_out = tf.nn.avg_pool(x, [2, 2], strides=[1, 1, 1, 1], padding='SAME')
>>> print(conv_out.shape, pool_out.shape)
(1, 3, 2, 1) (1, 3, 2, 1)
>>> tf.reshape(conv_out, [3, 2]).numpy()  # conv2d takes account of padding
array([[1.5 , 0.75],
       [1.  , 0.5 ],
       [0.5 , 0.25]], dtype=float32)
>>> tf.reshape(pool_out, [3, 2]).numpy()  # avg_pool excludes padding
array([[1.5, 1.5],
       [1. , 1. ],
       [1. , 1. ]], dtype=float32)


"""
