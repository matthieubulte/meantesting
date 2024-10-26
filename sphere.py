import jax
import jax.numpy as jnp
from manifold import Manifold

class Sphere(Manifold):
    @classmethod
    def exp(cls, x, u):
        norm_u = jnp.linalg.norm(u)
        return jax.lax.cond(
            norm_u < 1e-10,
            lambda _: x,
            lambda _: jnp.cos(norm_u) * x + jnp.sin(norm_u) * u / norm_u,
            None,
        )
    
    @classmethod
    def log(cls, x, y):
        delta = y - x
        proj = proj_comp(x, delta)
        proj_norm = jnp.linalg.norm(proj)
        return jax.lax.cond(
            proj_norm < 1e-10,
            lambda _: jnp.zeros(2),
            lambda _: jnp.arccos(jnp.dot(x, y)) * proj / proj_norm,
            None,
        )


    @classmethod
    def dist(cls, x, y):
        return jnp.arccos(jnp.dot(x, y))


    @classmethod
    def frechet_mean(cls, X, _, max_iter=10):
        def body_fn(carry, _):
            (mu,) = carry
            v_mu_new = jax.vmap(lambda x: Sphere.log(mu, x))(X).mean(axis=0)
            return (Sphere.exp(mu, v_mu_new),), None

        initial_state = (X[0],)
        (mu_hat,), _ = jax.lax.scan(body_fn, initial_state, jnp.arange(max_iter))
        return mu_hat


def proj_comp(x, u):
    return u - jnp.dot(x, u) * x
