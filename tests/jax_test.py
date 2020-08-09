
from jax import grad, vmap
import jax.numpy as np

import numpy as _np

def tanh(x):
    y = np.exp(-2. * x)
    return (1. - y) / (1. + y)

def test1():

    d_tanh = grad(tanh)

    # x = _np.random.randn(10,2)
    x = _np.random.randn(10)
    y = tanh(x)
    print(y, type(y))

    # input()

    # print(d_tanh(1.0))
    # print(d_tanh(x))
    print(vmap(d_tanh)(x))
    # print(tanh(x))

    # print(vmap(grad(tanh))(x))


def test2():
    d_tanh = grad(tanh)

    # x = _np.random.randn(10,2)
    x = _np.random.randn(10, 2)
    print(x)

    f = lambda x:tanh(x.sum())
    df = grad(f)
    y = f(x)

    print(y, type(y))

    # input()

    # print(d_tanh(1.0))
    # print(d_tanh(x))
    print(vmap(df)(x))
    # print(tanh(x))


def main():
    # test1()
    test2()

if __name__ == '__main__':
    main()
