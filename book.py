import jax
import jax.numpy as jnp
from manifold import Manifold
import spider


def Book(k):
    Spider = spider.Spider(k)

    class _Book(Manifold):
        K = k

        @classmethod
        def dist(cls, x, y):
            return jnp.sqrt(Spider.dist(x[:2], y[:2]) ** 2 + jnp.abs(x[2] - y[2]) ** 2)

        @classmethod
        def frechet_mean(cls, x, _):
            return jnp.concatenate(
                [Spider.frechet_mean(x[:, :2], _), jnp.mean(x[:, 2:], axis=0)]
            )

        @classmethod
        def reflect(cls, x_refl, xs):
            k_refl = x_refl[0]
            ks = jnp.mod(xs[:, 0] + 1, cls.K + 1)
            v0, v1 = jnp.where(k_refl == cls.K, 0, k_refl), jnp.where(
                k_refl == cls.K, k_refl, k_refl + 1
            )
            m0, m1 = ks == v0, ks == v1
            ks = jnp.where(m0, v1, jnp.where(m1, v0, ks))
            return jnp.column_stack([ks, xs[:, 1], 2 * x_refl[2:] - xs[:, 2:]])

    _Book.K = k

    return _Book
