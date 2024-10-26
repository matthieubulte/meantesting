import jax
import jax.numpy as jnp
from manifold import Manifold

class BuresWasserstein(Manifold):
    @classmethod
    def exp(cls, A, B):
        return (jnp.eye(A.shape[0]) + B) @ A @ (jnp.eye(A.shape[0]) + B)
    
    @classmethod
    def log(cls, A, B):
        return T(A, B) - jnp.eye(B.shape[0])


    @classmethod
    def dist(cls, A, B):
        sqrt_A = sqrtm(A)
        return jnp.sqrt(jnp.trace(A + B - 2 * sqrtm(sqrt_A @ B @ sqrt_A)))

    @classmethod
    def frechet_mean(cls, X, key, learning_rate=0.01, n_iterations=None, decay_rate=0.001):
        n_iterations = n_iterations or X.shape[0]
        dim = X.shape[1]
        Id = jnp.eye(dim, dtype=X.dtype)

        def body_fn(carry, t):
            Sigma_t, key = carry
            eta_t = learning_rate / (1 + decay_rate * t)
            key, subkey = jax.random.split(key)
            K_t = X[jax.random.randint(subkey, (), 0, X.shape[0])]
            G_hat_t = (1 - eta_t) * Id + eta_t * T(Sigma_t, K_t)
            Sigma_t = sym(G_hat_t @ Sigma_t @ G_hat_t)
            return (Sigma_t, key), None

        initial_state = (Id, key)
        (Sigma_t, _), _ = jax.lax.scan(body_fn, initial_state, jnp.arange(n_iterations))

        return Sigma_t
        
def sym(A):
    return (A + A.T) / 2


def sqrtm(A):
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    trace, det = a + d, a * d - b * c
    eig1 = (trace / 2) + jnp.sqrt((trace / 2) ** 2 - det)
    eig2 = (trace / 2) - jnp.sqrt((trace / 2) ** 2 - det)
    sqrt_eig1, sqrt_eig2 = jnp.sqrt(eig1), jnp.sqrt(eig2)

    def b_zero():
        return jnp.array([[sqrt_eig1, 0], [0, sqrt_eig2]])

    def b_non_zero():
        v1 = jnp.array([eig1 - d, b]) / jnp.linalg.norm(jnp.array([eig1 - d, b]))
        v2 = jnp.array([eig2 - d, b]) / jnp.linalg.norm(jnp.array([eig2 - d, b]))
        P = jnp.column_stack((v1, v2))
        sqrt_D = jnp.diag(jnp.array([sqrt_eig1, sqrt_eig2]))
        return P @ sqrt_D @ P.T

    return jax.lax.cond(b != 0, b_non_zero, b_zero)


def T(A, B):
    sqrt_A = sqrtm(A)
    sqrt_A_inv = jnp.linalg.inv(sqrt_A)
    return sqrt_A_inv @ sqrtm(sym(sqrt_A @ B @ sqrt_A)) @ sqrt_A_inv
