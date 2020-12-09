import numpy as np
# from fxpmath import sum as sumf
from fxpmath import Fxp


def format(x, n_word=16, n_int=None):
    return Fxp(x).like(Fxp(None, n_word=n_word, n_int=n_int))


def fxp2int(*args):
    max_n_frac = max(map(lambda arg: arg.n_frac, args))
    shifts = list(map(lambda arg: max_n_frac - arg.n_frac, args))
    args = list(map(lambda arg, shift: arg << shift, args, shifts))
    # _args = []
    # print('========')
    # for arg, shift in zip(args, shifts):
    #     # print('--------------')
    #     # print(arg.info(), shift)
    #     _arg = arg << shift
    #     _args.append(_arg)
    # args = _args
    for arg in args:
        arg.resize(arg.signed, n_frac=0)
    return list(map(lambda arg: arg.astype(int), args)), max_n_frac


# TODO: オーバーフロー
def matmul(a, b):
    (a_int, b_int), shift = fxp2int(a, b)
    z_int = np.matmul(a_int, b_int)
    z = Fxp(z_int.astype(np.float))
    z.resize(n_frac=shift * 2)
    return z


# TODO: オーバーフロー
def square_sum(a):
    x = np.inner(a[..., 0], a[..., 0])
    y = np.inner(a[..., 1], a[..., 1])
    x = format(x, n_word=16, n_int=5)
    y = format(x, n_word=16, n_int=5)
    z = x + y
    z = format(z, n_word=16, n_int=5)
    # print('z: {}', z.info())
    return sum(z, axis=-1)


# TODO: オーバーフロー
def cross(a, b):
    (a_int, b_int), shift = fxp2int(a, b)
    z_int = np.cross(a_int, b_int)
    z = Fxp(z_int.astype(np.float))
    z.resize(n_frac=shift * 2)
    return z


def concatenate(args, axis=None):
    args_int, shift = fxp2int(*args)
    z_int = np.concatenate(args_int, axis=axis)
    z = Fxp(z_int.astype(np.float))
    z.resize(n_frac=shift)
    return z


# TODO: オーバーフロー！！
def sum(a, axis=None):
    a_int, shift = fxp2int(a)
    z_int = np.sum(a_int[0], axis=axis)
    z = Fxp(z_int.astype(float))
    z.resize(n_frac=shift)
    return z
