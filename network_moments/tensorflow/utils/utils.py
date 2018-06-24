import numpy as np
import tensorflow as tf


def epsilon(dtype=tf.float32):
    '''Machine epsilon for a specific numpy or tensorflow dtype.'''
    if isinstance(dtype, tf.DType):
        dtype = dtype.min.dtype
    return np.finfo(dtype).eps


def mul_diag(A, vec):
    '''Batch support for matrix diag(vector) product.

    Args:
        A: General matrix of size (M, Size).
        vec: Vector of size (Batch, Size).

    Returns:
        The result of multiplying A with diag(vec) (Batch, M).
    '''
    with tf.name_scope('mul_diag'):
        return A * tf.expand_dims(vec, -2)


def matmul(A, B, transpose_a=False, transpose_b=False):
    '''Batch support for matrix matrix product.

    Args:
        A: General matrix of size (A_Batch, M, X).
        B: General matrix of size (B_Batch, X, N).
        transpose_a: Whether A is transposed (A_Batch, X, M).
        transpose_b: Whether B is transposed (B_Batch, N, X).

    Returns:
        The result of multiplying A with B (A_Batch, B_Batch, M, N).
        Works more efficiently if B_Batch is empty.
    '''
    Andim = len(A.shape)
    Bndim = len(B.shape)
    if Andim == Bndim:
        return tf.matmul(A, B, transpose_a=transpose_a,
                         transpose_b=transpose_b)  # faster than tensordot
    with tf.name_scope('matmul'):
        a_index = Andim - (2 if transpose_a else 1)
        b_index = Bndim - (1 if transpose_b else 2)
        AB = tf.tensordot(A, B, axes=[a_index, b_index])
        if Bndim > 2:  # only if B is batched, rearrange the axes
            A_Batch = np.arange(Andim - 2)
            M = len(A_Batch)
            B_Batch = (M + 1) + np.arange(Bndim - 2)
            N = (M + 1) + len(B_Batch)
            perm = np.concatenate((A_Batch, B_Batch, [M, N]))
            AB = tf.transpose(AB, perm)
        return AB


def outer(vec):
    '''Batch support for the outer product of a vector with itself.

    Args:
        vec: A vector of size (Batch, Size).

    Returns:
        The outer product of vec with itself (Batch, Size, Size).
    '''
    with tf.name_scope('outer_product'):
        vec = tf.expand_dims(vec, axis=-1)
        return matmul(vec, vec, transpose_b=True)


def normalize(vec, norm=1.0):
    '''Normalize a batch of vectors.

    Args:
        vec: A vector of size (Batch, Size).
        norm: The new norm (default is 1.0).

    Returns:
        The normalized vector.
    '''
    new_norm = norm / np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec * new_norm


def jacobian(outputs, inputs, at):
    '''Compute the Jacobian matrix of outputs w.r.t. inputs at a given input.

    Args:
        outputs: Multivariate output tensor.
        inputs: Multivariate input tensor.
        at: Batch of points at which to compute the Jacobian (Batch, *Size)
            Batch must be at least 1.

    Returns
        The Jacobian of the model at all the points in `at`.
    '''
    return linearize(outputs, inputs, at, jacobian_only=True)


def _masks_batches(shape, batch_size):
    '''Batches iterator over all possible masks of the given shape.

    A mask is a numpy.ndarray of shape `shape` of all zeros except
    for a single position it is one. It is useful to get those masks
    in batches instead of getting them one by one.

    Args:
        shape: The shape of each mask.
        batch_size: How many masks to return in each iteration.

    Returns:
        A batch of masks of shape [batch_size, *shape].
    '''
    num_rows = np.prod(shape)
    if num_rows < batch_size:
        batch_size = num_rows
    _mask = np.zeros((batch_size, *shape))
    mask = _mask.reshape(batch_size, -1)
    num_batches = -(-num_rows // batch_size)
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_rows)
        if end - start < batch_size:
            batch_size = end - start
            _mask = np.zeros((batch_size, *shape))
            mask = _mask.reshape(batch_size, -1)
        mask[:, start:end] = np.eye(batch_size)
        yield _mask
        mask[:, start:end] = 0


def linearize(outputs, inputs, at, jacobian_only=False, batch_size=5000):
    '''Approximate the output of a model at a given point with an affine function.

    The first order Taylor decomposition of `outputs` at the point `inputs=at`
    is an affine transformation `f(x) = A * x + b` such that `outputs = f(at)`.
    A is the Jacobian of `outputs` w.r.t. `inputs` at the point `at`
    and b is simply `b = outputs - A * x`.

    follow this link for updates on implementing this function in the graph:
    https://github.com/tensorflow/tensorflow/issues/675

    Args:
        outputs: Multivariate output tensor.
        inputs: Multivariate input tensor.
        at: Batch of points (Batch, *Size) at which to compute the Jacobian A
            Batch must be at least 1.
        jacobian_only: Whether to return only A.
        batch_size: The number of rows in the Jacobian to compute at once.

    Returns
        The matrix A and the bias vector b.
    '''
    # construct gradient mask only once
    shape = outputs.shape.as_list()
    if hasattr(outputs, 'gradient_mask'):
        grad_mask = outputs.gradient_mask
    else:
        grad_mask = tf.placeholder(inputs.dtype, shape=shape,
                                   name=outputs.op.name + '_gradient_mask')
        outputs.gradient_mask = grad_mask
        outputs.gradient = {}

    # construct gradient tensor only once
    if inputs in outputs.gradient:
        grad = outputs.gradient[inputs]
    else:
        name = outputs.op.name + '_gradient_' + inputs.op.name
        grad = tf.gradients(outputs * grad_mask, inputs, name=name)[0]
        outputs.gradient[inputs] = grad

    # compute the Jacobian row-batch by row-batch for all points
    jacobians_list = []
    sess = tf.get_default_session()
    for i in range(at.shape[0]):
        # extract the Jacobian matrix for a single point
        partials_list = []
        point = at[i:i + 1, :]
        shape = outputs.shape.as_list()[1:]
        repeated_point = point
        for mask in _masks_batches(shape, batch_size):
            # repeat the point according to the mask's batch_size
            batch_size = mask.shape[0]
            if repeated_point.shape[0] < batch_size:
                repeated_point = np.vstack([point] * batch_size)
            if repeated_point.shape[0] > batch_size:
                repeated_point = repeated_point[:batch_size, :]
            feed = {grad_mask: mask, inputs: repeated_point}
            partial = sess.run(grad, feed_dict=feed)
            partials_list.append(partial)
        jacobian = np.vstack(partials_list)

        # reshape it as a matrix
        jacobians_list.append(jacobian.reshape(jacobian.shape[0], -1))

    # stack Jacobians
    A = np.stack(jacobians_list)
    if jacobian_only:
        return A

    b = sess.run(outputs, feed_dict={inputs: at}).reshape(at.shape[0], -1)
    b -= np.matmul(A, at.reshape(at.shape[0], -1, 1))[..., 0]
    return A, b


def rand_from_eigen(eigen):
    '''Construct a random matrix with given the eigenvalues.

    To construct such a matrix form the eigenvalue decomposition,
    (i.e. U * Sigma * U.t()), we need to find a unitary matrix U
    and Sigma is the diagonal matrix of the eigenvalues `eigen`.
    The matrix U can be the unitary matrix Q from
    the QR-decomposition of a randomly generated matrix.

    Args:
        eigen: A vector of size (Batch, Size).

    Returns:
        A random matrix of size (Batch, Size, Size).
    '''
    size = eigen.shape[-1]
    Q = np.linalg.qr(np.random.randn(size, size).astype(eigen.dtype))[0]
    return (Q * np.expand_dims(eigen, -2)).dot(Q.T)


def rand_definite(size, batch=None, norm=None,
                  positive=True, semi=False, dtype=None):
    '''Random definite matrix.

    A positive/negative definite matrix is a matrix
    with positive/negative eigenvalues, respectively.
    They are called semi-definite if the eigenvalues are allowed to be zeros.
    The eigenvalues are some random vector of unit norm.
    This vector is what control (positive vs. negative)
    and (semi-definite vs definite).
    We multiply this vector by the desired `norm`.

    Args:
        size: The output matrix is of size (`size`, `size`).
        batch: Number of matrices to generate.
        norm: The Frobenius norm of the output matrix.
        positive: Whether positive-definite or negative-definite.
        semi: Whether to construct semi-definite or definite matrix.
        dtype: The data type.

    Returns:
        Random definite matrices of size (`Batch`, `size`, `size`)
        and Frobenius norm `norm`.
    '''
    shape = size if batch is None else (batch, size)
    eigen = np.random.rand(shape).astype(dtype)
    if not semi:
        eigen = 1.0 - eigen
    if not positive:
        eigen = -eigen
    eigen = eigen if norm is None else normalize(eigen, norm)
    return rand_from_eigen(eigen)
