"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.util.tf_export import tf_export

"""`LinearOperator` coming from a [[nested] block] circulant matrix."""
__all__ = ["LinearOperatorCirculant", "LinearOperatorCirculant2D", "LinearOperatorCirculant3D"]
_FFT_OP = ...
_IFFT_OP = ...
def exponential_power_convolution_kernel(grid_shape, length_scale, power=..., divisor=..., zero_inflation=...):
  """Make an exponentiated convolution kernel.

  In signal processing, a [kernel]
  (https://en.wikipedia.org/wiki/Kernel_(image_processing)) `h` can be convolved
  with a signal `x` to filter its spectral content.

  This function makes a `d-dimensional` convolution kernel `h` of shape
  `grid_shape = [N0, N1, ...]`. For `n` a multi-index with `n[i] < Ni / 2`,

  ```h[n] = exp{sum(|n / (length_scale * grid_shape)|**power) / divisor}.```

  For other `n`, `h` is extended to be circularly symmetric. That is

  ```h[n0 % N0, ...] = h[(-n0) % N0, ...]```

  Since `h` is circularly symmetric and real valued, `H = FFTd[h]` is the
  spectrum of a symmetric (real) circulant operator `A`.

  #### Example uses

  ```
  # Matern one-half kernel, d=1.
  # Will be positive definite without zero_inflation.
  h = exponential_power_convolution_kernel(
      grid_shape=[10], length_scale=[0.1], power=1)
  A = LinearOperatorCirculant(
      tf.signal.fft(tf.cast(h, tf.complex64)),
      is_self_adjoint=True, is_positive_definite=True)

  # Gaussian RBF kernel, d=3.
  # Needs zero_inflation since `length_scale` is long enough to cause aliasing.
  h = exponential_power_convolution_kernel(
      grid_shape=[10, 10, 10], length_scale=[0.1, 0.2, 0.2], power=2,
      zero_inflation=0.15)
  A = LinearOperatorCirculant3D(
      tf.signal.fft3d(tf.cast(h, tf.complex64)),
      is_self_adjoint=True, is_positive_definite=True)
  ```

  Args:
    grid_shape: Length `d` (`d` in {1, 2, 3}) list-like of Python integers. The
      shape of the grid on which the convolution kernel is defined.
    length_scale: Length `d` `float` `Tensor`. The scale at which the kernel
      decays in each direction, as a fraction of `grid_shape`.
    power: Scalar `Tensor` of same `dtype` as `length_scale`, default `2`.
      Higher (lower) `power` results in nearby points being more (less)
      correlated, and far away points being less (more) correlated.
    divisor: Scalar `Tensor` of same `dtype` as `length_scale`. The slope of
      decay of `log(kernel)` in terms of fractional grid points, along each
      axis, at `length_scale`, is `power/divisor`. By default, `divisor` is set
      to `power`. This means, by default, `power=2` results in an exponentiated
      quadratic (Gaussian) kernel, and `power=1` is a Matern one-half.
    zero_inflation: Scalar `Tensor` of same `dtype` as `length_scale`, in
      `[0, 1]`. Let `delta` be the Kronecker delta. That is,
      `delta[0, ..., 0] = 1` and all other entries are `0`. Then
      `zero_inflation` modifies the return value via
      `h --> (1 - zero_inflation) * h + zero_inflation * delta`. This may be
      needed to ensure a positive definite kernel, especially if `length_scale`
      is large enough for aliasing and `power > 1`.

  Returns:
    `Tensor` of shape `grid_shape` with same `dtype` as `length_scale`.
  """
  ...

class _BaseLinearOperatorCirculant(linear_operator.LinearOperator):
  """Base class for circulant operators.  Not user facing.

  `LinearOperator` acting like a [batch] [[nested] block] circulant matrix.
  """
  def __init__(self, spectrum, block_depth, input_output_dtype=..., is_non_singular=..., is_self_adjoint=..., is_positive_definite=..., is_square=..., parameters=..., name=...) -> None:
    r"""Initialize an `_BaseLinearOperatorCirculant`.

    Args:
      spectrum:  Shape `[B1,...,Bb] + N` `Tensor`, where `rank(N) in {1, 2, 3}`.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.  Type can be different than `input_output_dtype`
      block_depth:  Python integer, either 1, 2, or 3.  Will be 1 for circulant,
        2 for block circulant, and 3 for nested block circulant.
      input_output_dtype: `dtype` for input/output.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      parameters: Python `dict` of parameters used to instantiate this
        `LinearOperator`.
      name:  A name to prepend to all ops created by this class.

    Raises:
      ValueError:  If `block_depth` is not an allowed value.
      TypeError:  If `spectrum` is not an allowed type.
    """
    ...
  
  @property
  def block_depth(self): # -> Unknown:
    """Depth of recursively defined circulant blocks defining this `Operator`.

    With `A` the dense representation of this `Operator`,

    `block_depth = 1` means `A` is symmetric circulant.  For example,

    ```
    A = |w z y x|
        |x w z y|
        |y x w z|
        |z y x w|
    ```

    `block_depth = 2` means `A` is block symmetric circulant with symmetric
    circulant blocks.  For example, with `W`, `X`, `Y`, `Z` symmetric circulant,

    ```
    A = |W Z Y X|
        |X W Z Y|
        |Y X W Z|
        |Z Y X W|
    ```

    `block_depth = 3` means `A` is block symmetric circulant with block
    symmetric circulant blocks.

    Returns:
      Python `integer`.
    """
    ...
  
  def block_shape_tensor(self): # -> Tensor | Any:
    """Shape of the block dimensions of `self.spectrum`."""
    ...
  
  @property
  def block_shape(self): # -> Any:
    ...
  
  @property
  def spectrum(self): # -> Tensor | Any | None:
    ...
  
  def convolution_kernel(self, name=...): # -> SparseTensor | IndexedSlices | Tensor | Any:
    """Convolution kernel corresponding to `self.spectrum`.

    The `D` dimensional DFT of this kernel is the frequency domain spectrum of
    this operator.

    Args:
      name:  A name to give this `Op`.

    Returns:
      `Tensor` with `dtype` `self.dtype`.
    """
    ...
  
  def assert_hermitian_spectrum(self, name=...): # -> Any | None:
    """Returns an `Op` that asserts this operator has Hermitian spectrum.

    This operator corresponds to a real-valued matrix if and only if its
    spectrum is Hermitian.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Op` that asserts this operator has Hermitian spectrum.
    """
    ...
  


@tf_export("linalg.LinearOperatorCirculant")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a circulant matrix.

  This operator acts like a circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of circulant matrices

  Circulant means the entries of `A` are generated by a single vector, the
  convolution kernel `h`: `A_{mn} := h_{m-n mod N}`.  With `h = [w, x, y, z]`,

  ```
  A = |w z y x|
      |x w z y|
      |y x w z|
      |z y x w|
  ```

  This means that the result of matrix multiplication `v = Au` has `Lth` column
  given circular convolution between `h` with the `Lth` column of `u`.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.  Define the discrete Fourier transform (DFT) and its inverse by

  ```
  DFT[ h[n] ] = H[k] := sum_{n = 0}^{N - 1} h_n e^{-i 2pi k n / N}
  IDFT[ H[k] ] = h[n] = N^{-1} sum_{k = 0}^{N - 1} H_k e^{i 2pi k n / N}
  ```

  From these definitions, we see that

  ```
  H[0] = sum_{n = 0}^{N - 1} h_n
  H[1] = "the first positive frequency"
  H[N - 1] = "the first negative frequency"
  ```

  Loosely speaking, with `*` element-wise multiplication, matrix multiplication
  is equal to the action of a Fourier multiplier: `A u = IDFT[ H * DFT[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT[u]` be the `[N, R]`
  matrix with `rth` column equal to the DFT of the `rth` column of `u`.
  Define the `IDFT` similarly.
  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT[ H * (DFT[u])_r ]```

  #### Operator properties deduced from the spectrum.

  Letting `U` be the `kth` Euclidean basis vector, and `U = IDFT[u]`.
  The above formulas show that`A U = H_k * U`.  We conclude that the elements
  of `H` are the eigenvalues of this operator.   Therefore

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N]`.  We say that `H` is a Hermitian spectrum
  if, with `%` meaning modulus division,

  ```H[..., n % N] = ComplexConjugate[ H[..., (-n) % N] ]```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  #### Example of a self-adjoint positive definite operator

  ```python
  # spectrum is real ==> operator is self-adjoint
  # spectrum is positive ==> operator is positive definite
  spectrum = [6., 4, 2]

  operator = LinearOperatorCirculant(spectrum)

  # IFFT[spectrum]
  operator.convolution_kernel()
  ==> [4 + 0j, 1 + 0.58j, 1 - 0.58j]

  operator.to_dense()
  ==> [[4 + 0.0j, 1 - 0.6j, 1 + 0.6j],
       [1 + 0.6j, 4 + 0.0j, 1 - 0.6j],
       [1 - 0.6j, 1 + 0.6j, 4 + 0.0j]]
  ```

  #### Example of defining in terms of a real convolution kernel

  ```python
  # convolution_kernel is real ==> spectrum is Hermitian.
  convolution_kernel = [1., 2., 1.]]
  spectrum = tf.signal.fft(tf.cast(convolution_kernel, tf.complex64))

  # spectrum is Hermitian ==> operator is real.
  # spectrum is shape [3] ==> operator is shape [3, 3]
  # We force the input/output type to be real, which allows this to operate
  # like a real matrix.
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

  operator.to_dense()
  ==> [[ 1, 1, 2],
       [ 2, 1, 1],
       [ 1, 2, 1]]
  ```

  #### Example of Hermitian spectrum

  ```python
  # spectrum is shape [3] ==> operator is shape [3, 3]
  # spectrum is Hermitian ==> operator is real.
  spectrum = [1, 1j, -1j]

  operator = LinearOperatorCirculant(spectrum)

  operator.to_dense()
  ==> [[ 0.33 + 0j,  0.91 + 0j, -0.24 + 0j],
       [-0.24 + 0j,  0.33 + 0j,  0.91 + 0j],
       [ 0.91 + 0j, -0.24 + 0j,  0.33 + 0j]
  ```

  #### Example of forcing real `dtype` when spectrum is Hermitian

  ```python
  # spectrum is shape [4] ==> operator is shape [4, 4]
  # spectrum is real ==> operator is self-adjoint
  # spectrum is Hermitian ==> operator is real
  # spectrum has positive real part ==> operator is positive-definite.
  spectrum = [6., 4, 2, 4]

  # Force the input dtype to be float32.
  # Cast the output to float32.  This is fine because the operator will be
  # real due to Hermitian spectrum.
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

  operator.shape
  ==> [4, 4]

  operator.to_dense()
  ==> [[4, 1, 0, 1],
       [1, 4, 1, 0],
       [0, 1, 4, 1],
       [1, 0, 1, 4]]

  # convolution_kernel = tf.signal.ifft(spectrum)
  operator.convolution_kernel()
  ==> [4, 1, 0, 1]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.

  References:
    Toeplitz and Circulant Matrices - A Review:
      [Gray, 2006](https://www.nowpublishers.com/article/Details/CIT-006)
      ([pdf](https://ee.stanford.edu/~gray/toeplitz.pdf))
  """
  def __init__(self, spectrum, input_output_dtype=..., is_non_singular=..., is_self_adjoint=..., is_positive_definite=..., is_square=..., name=...) -> None:
    r"""Initialize an `LinearOperatorCirculant`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N]` `Tensor`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N]` `Tensor`.  Allowed dtypes: `float16`,
        `float32`, `float64`, `complex64`, `complex128`.  Type can be different
        than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    ...
  


@tf_export("linalg.LinearOperatorCirculant2D")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant2D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a block circulant matrix.

  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of block circulant matrices

  If `A` is block circulant, with block sizes `N0, N1` (`N0 * N1 = N`):
  `A` has a block circulant structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` circulant matrix.

  For example, with `W`, `X`, `Y`, `Z` each circulant,

  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```

  Note that `A` itself will not in general be circulant.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.

  If `H.shape = [N0, N1]`, (`N0 * N1 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT2[ H DFT2[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT2[u]` be the
  `[N0, N1, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, R]` and taking
  a two dimensional DFT across the first two dimensions.  Let `IDFT2` be the
  inverse of `DFT2`.  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT2[ H * (DFT2[u])_r ]```

  #### Operator properties deduced from the spectrum.

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N0, N1]`, we say that `H` is a Hermitian
  spectrum if, with `%` indicating modulus division,

  ```
  H[..., n0 % N0, n1 % N1] = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1 ].
  ```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  ### Example of a self-adjoint positive definite operator

  ```python
  # spectrum is real ==> operator is self-adjoint
  # spectrum is positive ==> operator is positive definite
  spectrum = [[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]]

  operator = LinearOperatorCirculant2D(spectrum)

  # IFFT[spectrum]
  operator.convolution_kernel()
  ==> [[5.0+0.0j, -0.5-.3j, -0.5+.3j],
       [-1.5-.9j,        0,        0],
       [-1.5+.9j,        0,        0]]

  operator.to_dense()
  ==> Complex self adjoint 9 x 9 matrix.
  ```

  #### Example of defining in terms of a real convolution kernel,

  ```python
  # convolution_kernel is real ==> spectrum is Hermitian.
  convolution_kernel = [[1., 2., 1.], [5., -1., 1.]]
  spectrum = tf.signal.fft2d(tf.cast(convolution_kernel, tf.complex64))

  # spectrum is shape [2, 3] ==> operator is shape [6, 6]
  # spectrum is Hermitian ==> operator is real.
  operator = LinearOperatorCirculant2D(spectrum, input_output_dtype=tf.float32)
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  def __init__(self, spectrum, input_output_dtype=..., is_non_singular=..., is_self_adjoint=..., is_positive_definite=..., is_square=..., name=...) -> None:
    r"""Initialize an `LinearOperatorCirculant2D`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N0, N1]` `Tensor` with `N0*N1 = N`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N0, N1]` `Tensor`.  Allowed dtypes:
        `float16`, `float32`, `float64`, `complex64`, `complex128`.
        Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    ...
  


@tf_export("linalg.LinearOperatorCirculant3D")
@linear_operator.make_composite_tensor
class LinearOperatorCirculant3D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a nested block circulant matrix.

  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of block circulant matrices

  If `A` is nested block circulant, with block sizes `N0, N1, N2`
  (`N0 * N1 * N2 = N`):
  `A` has a block structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` block circulant matrix.

  For example, with `W`, `X`, `Y`, `Z` each block circulant,

  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```

  Note that `A` itself will not in general be circulant.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.

  If `H.shape = [N0, N1, N2]`, (`N0 * N1 * N2 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT3[ H DFT3[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT3[u]` be the
  `[N0, N1, N2, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, N2, R]` and
  taking a three dimensional DFT across the first three dimensions.  Let `IDFT3`
  be the inverse of `DFT3`.  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT3[ H * (DFT3[u])_r ]```

  #### Operator properties deduced from the spectrum.

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N0, N1, N2]`, we say that `H` is a Hermitian
  spectrum if, with `%` meaning modulus division,

  ```
  H[..., n0 % N0, n1 % N1, n2 % N2]
    = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1, (-n2) % N2] ].
  ```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  ### Examples

  See `LinearOperatorCirculant` and `LinearOperatorCirculant2D` for examples.

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """
  def __init__(self, spectrum, input_output_dtype=..., is_non_singular=..., is_self_adjoint=..., is_positive_definite=..., is_square=..., name=...) -> None:
    """Initialize an `LinearOperatorCirculant`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N0, N1, N2]` `Tensor`
    with `N0*N1*N2 = N`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N0, N1, N2]` `Tensor`.  Allowed dtypes:
        `float16`, `float32`, `float64`, `complex64`, `complex128`.
        Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    ...
  


