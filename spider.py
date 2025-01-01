import jax
import jax.numpy as jnp
from manifold import Manifold, dist_sq_arr


def Spider(k):
    class _Spider(Manifold):
        K = k

        @classmethod
        def dist(cls, x, y):
            return jnp.where(x[0] == y[0], jnp.abs(x[1] - y[1]), jnp.abs(x[1] + y[1]))

        @classmethod
        def frechet_mean(cls, x, _):
            n = x.shape[0]

            @jax.jit
            def compute_branch_stats(i):
                in_branch = 2.0 * (x[:, 0] == i) - 1
                x_star = jnp.dot(in_branch, x[:, 1]) / n
                x_star_fixed = jnp.clip(x_star, 0, 1)
                loss = dist_sq_arr(cls, jnp.array([i, x_star_fixed]), x).sum()
                return loss, x_star_fixed

            losses_and_x_stars = jax.vmap(compute_branch_stats)(jnp.arange(cls.K))
            losses, x_stars = losses_and_x_stars[0], losses_and_x_stars[1]
            best_branch = jnp.argmin(losses)
            return jnp.array([best_branch, x_stars[best_branch]])

    return _Spider
