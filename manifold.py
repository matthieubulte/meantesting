from typing import Protocol
import jax


class Manifold(Protocol):
    @classmethod
    def exp(cls, base_point, tangent_vector): ...

    @classmethod
    def log(cls, base_point, target_point): ...

    @classmethod
    def dist(cls, base_point, target_point): ...

    @classmethod
    def frechet_mean(cls, X, key, **kwargs): ...


def mirror_vecs(M, mu, ys):
    return jax.vmap(lambda y: M.exp(mu, -M.log(mu, y)))(ys)


def geodesic(M, x, y, delta):
    return M.exp(x, delta * M.log(x, y))
