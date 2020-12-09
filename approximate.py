import numpy as np


def log(x):
    """
    Mitchell対数

    Parameters
    ----------
    x : Integer

    Returns
    -------
    result : double
        xとAのそれぞれのベクトルと行列の積
    """
    pass


def original_multiply(x, y):
    pass


def _multiply(x, y):
    if x > 0:
        xx = np.floor(np.log2(x))
        xxp = 1
    elif x < 0:
        xx = np.floor(np.log2(-x))
        xxp = -1
    else:
        xx = 1
        xxp = 0

    if y > 0:
        yy = np.floor(np.log2(y))
        yyp = 1
    elif y < 0:
        yy = np.floor(np.log2(-y))
        yyp = -1
    else:
        yy = 1
        yyp = 0

    # carry 0
    if x * xxp * np.power(2.0, -xx) + y * yyp * np.power(2.0, -yy) < 3:
        return yyp * np.power(2.0, yy) * x \
             + xxp * np.power(2.0, xx) * y \
             - xxp * yyp * 2.0 ** (xx + yy)
    # carry 1
    else:
        return yyp * np.power(2.0, yy + 1) * x \
             + xxp * np.power(2.0, xx + 1) * y \
             - xxp * yyp * np.power(2.0, xx + yy + 2)


multiply = np.frompyfunc(_multiply, 2, 1)


def matmul(x, A):
    """
    numpy.matmulの近似版だが、完全に同じではない。

    Parameters
    ----------
    x : (k, 2)
        要素数2のベクトルのarray

    A : (n, m, 2, 2)
        2x2行列のarray

    Returns
    -------
    result : (n, m, k, 2)
        xとAのそれぞれのベクトルと行列の積
    """
    # print(x.shape)
    # print(A.shape)

    A_dim = A.ndim - 2
    x_dim = x.ndim - 1

    xx = np.tile(x[..., np.newaxis], A.shape[-1])
    xx = xx.reshape((1,) * A_dim + xx.shape)
    xx = np.tile(xx, A.shape[:-2] + (1,) * (x_dim + 2))

    AA = A.reshape(A.shape[:-2] + (1,) * x_dim + A.shape[-2:])
    AA = np.tile(AA, (1,) * A_dim + x.shape[:-1] + (1,) * 2)

    xA = np.stack([xx, AA], axis=-2)
    return multiply.reduce(xA, axis=-2).sum(axis=-2).astype(np.float64)


def norm(x, axis=None):
    return np.sqrt(np.sum(multiply(x, x), axis=axis).astype(np.float64))


def square_norm(x, axis=None):
    return np.sum(multiply(x, x), axis=axis).astype(np.float64)


def cross(x, y):
    """
    numpy.crossの近似版だが、完全に同じではない。

    Parameters
    ----------
    x : (k, 2)

    y : (k, 2)

    Returns
    -------
    result : (k)
    """
    return np.subtract.reduce(multiply.reduce(np.stack([x, y[..., (1, 0)]]), axis=0), axis=-1).astype(np.float64)
