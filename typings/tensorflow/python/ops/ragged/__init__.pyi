"""
This type stub file was generated by pyright.
"""

"""Ragged Tensors.

This package defines ops for manipulating ragged tensors (`tf.RaggedTensor`),
which are tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the `tf.RaggedTensor` class documentation
and the [Ragged Tensor Guide](/guide/ragged_tensor).
"""
