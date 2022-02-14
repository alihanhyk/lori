import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

jax.config.update("jax_platform_name", "cpu")


@jax.jit
def softmax(x, y):
    return logsumexp(np.stack((x, y)), axis=0)


@jax.jit
def softmin(x, y):
    return -logsumexp(np.stack((-x, -y)), axis=0)


###

r_true = np.array([[0, 1], [-1, 0]])
r_true_max = np.array([5, np.inf])

alp0, eps0 = np.log(1 / 0.1 - 1) / 0.1, 0.1
alp1, eps1 = np.log(1 / 0.1 - 1) / 0.1, 0.1


@jax.jit
def p_det(x, u):
    x1_0 = x[0] + 0.003 * x[0] * np.log(1000 / (1e-6 + x[0])) - 0.15 * x[0] * u
    x1_1 = x[1] + 1.2 - 0.15 * x[1] - 0.4 * x[1] * u
    return np.array([x1_0, x1_1])


@jax.jit
def p(x, u, key):
    return p_det(x, u) + 0.5 * jax.random.normal(key, shape=(2,))


@jax.jit
def p0(key):
    x0_0 = 30 + 5 * jax.random.normal(key)
    x0_1 = 8
    return np.array([x0_0, x0_1])


@jax.jit
def pref2(del0, del1):
    prob = 1 / (1 + np.exp(alp0 * (eps0 - del0)))
    prob = prob + (
        1
        - 1 / (1 + np.exp(alp0 * (eps0 - del0)))
        - 1 / (1 + np.exp(alp0 * (eps0 + del0)))
    ) * 1 / (1 + np.exp(alp1 * (eps1 - del1)))
    prob = (
        prob
        + (
            1
            - 1 / (1 + np.exp(alp0 * (eps0 - del0)))
            - 1 / (1 + np.exp(alp0 * (eps0 + del0)))
        )
        * (
            1
            - 1 / (1 + np.exp(alp1 * (eps1 - del1)))
            - 1 / (1 + np.exp(alp1 * (eps1 + del1)))
        )
        / 2
    )
    return prob


@jax.jit
def pref2_long(del0, del1, eps0, eps1):
    prob = 1 / (1 + np.exp(eps0 - del0))
    prob = prob + (
        1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0))
    ) * 1 / (1 + np.exp(eps1 - del1))
    prob = (
        prob
        + (1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0)))
        * (1 - 1 / (1 + np.exp(eps1 - del1)) - 1 / (1 + np.exp(eps1 + del1)))
        / 2
    )
    return prob


@jax.jit
def pref2_long1(del0, del1, eps0, eps1):
    prob = 1 / (1 + np.exp(eps0 - del0))
    prob = prob + (
        1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0))
    ) * 1 / (1 + np.exp(eps1 - del1))
    return prob


@jax.jit
def pref2_long1_equiv(del0, del1, eps0, eps1):
    return (1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0))) * (
        1 - 1 / (1 + np.exp(eps1 - del1)) - 1 / (1 + np.exp(eps1 + del1))
    )


@jax.jit
def pref2_long2(del0, del1, del2, eps0, eps1, eps2):
    prob = 1 / (1 + np.exp(eps0 - del0))
    prob = prob + (
        1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0))
    ) * 1 / (1 + np.exp(eps1 - del1))
    prob = prob + (
        1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0))
    ) * (1 - 1 / (1 + np.exp(eps1 - del1)) - 1 / (1 + np.exp(eps1 + del1))) * 1 / (
        1 + np.exp(eps2 - del2)
    )
    return prob


@jax.jit
def pref2_long2_equiv(del0, del1, del2, eps0, eps1, eps2):
    return (
        (1 - 1 / (1 + np.exp(eps0 - del0)) - 1 / (1 + np.exp(eps0 + del0)))
        * (1 - 1 / (1 + np.exp(eps1 - del1)) - 1 / (1 + np.exp(eps1 + del1)))
        * (1 - 1 / (1 + np.exp(eps2 - del2)) - 1 / (1 + np.exp(eps2 + del2)))
    )


###

state_n = 100
state_x0_min = 0
state_x0_max = 40
state_x1_min = 0
state_x1_max = 10

state_vals = np.array(
    [
        [[x0, x1] for x1 in np.linspace(state_x1_min, state_x1_max, state_n)]
        for x0 in np.linspace(state_x0_min, state_x0_max, state_n)
    ]
)
state_vals = state_vals.reshape(-1, 2)

_get_R0 = lambda r, x, u: r @ x
_get_R0 = jax.vmap(_get_R0, in_axes=(None, None, 0))
_get_R1 = lambda r, x: _get_R0(r, x, np.arange(2))
_get_R1 = jax.vmap(_get_R1, in_axes=(None, 0))
get_R = lambda r: _get_R1(r, state_vals)
get_R = jax.jit(get_R)
R_true, R_true_max = get_R(r_true), r_true_max

_get_T0 = lambda x, u, x1: np.sum(-(((x1 - p_det(x, u)) / 0.5) ** 2))
_get_T0 = jax.vmap(_get_T0, in_axes=(None, None, 0))
_get_T1 = lambda x, u: np.exp(
    _get_T0(x, u, state_vals) - logsumexp(_get_T0(x, u, state_vals))
)
_get_T1 = jax.vmap(_get_T1, in_axes=(None, 0))
_get_T2 = lambda x: _get_T1(x, np.arange(2))
_get_T2 = jax.jit(jax.vmap(_get_T2))
T = _get_T2(state_vals)


def _interp(x):
    i0 = np.maximum(
        0, np.minimum(1, (x[0] - state_x0_min) / (state_x0_max - state_x0_min))
    )
    i0 = np.round(i0 * (state_n - 1)).astype(int)
    i1 = np.maximum(
        0, np.minimum(1, (x[1] - state_x1_min) / (state_x1_max - state_x1_min))
    )
    i1 = np.round(i1 * (state_n - 1)).astype(int)
    return i0, i1


interp = lambda x, q: q.reshape(state_n, state_n, 2, 2)[_interp(x)]
interp = jax.jit(interp)
interp_single = lambda x, q: q.reshape(state_n, state_n, 2)[_interp(x)]
interp_single = jax.jit(interp_single)

###

lex_argmax = lambda q: np.lexsort(np.flip(q.T, axis=0))[-1]
lex_argmax = jax.jit(lex_argmax)
lex_max = lambda q: q[lex_argmax(q)]
lex_max = jax.jit(lex_max)
lex_maxmap = jax.vmap(lex_max)
lex_maxmap = jax.jit(lex_maxmap)


def _solve(arg0, arg1):
    (q, gmm, R, R_max, T), _ = arg0, arg1
    v = lex_maxmap(q)
    q = np.minimum(
        R_max / (1 - gmm),
        R + np.minimum(R_max / (1 - gmm), gmm * np.tensordot(T, v, axes=1)),
    )
    return (q, gmm, R, R_max, T), None


@jax.jit
def solve(gmm, R, R_max, T):
    q = np.zeros(R.shape)
    (q, *_), _ = jax.lax.scan(_solve, (q, gmm, R, R_max, T), np.arange(100))
    return q


def _solve_single(arg0, arg1):
    (q, gmm, R, T), _ = arg0, arg1
    v = np.amax(q, axis=-1)
    q = R + gmm * np.tensordot(T, v, axes=1)
    return (q, gmm, R, T), None


@jax.jit
def solve_single(gmm, R, T):
    q = np.zeros(R.shape)
    (q, *_), _ = jax.lax.scan(_solve_single, (q, gmm, R, T), np.arange(100))
    return q


@jax.jit
def pi(x, q, eps, key):
    u = lex_argmax(interp(x, q))
    u = np.where(jax.random.uniform(key) < (1 - eps) + eps / 2, u, 1 - u)
    return u


@jax.jit
def pi_single(x, q, eps, key):
    u = np.argmax(interp_single(x, q))
    u = np.where(jax.random.uniform(key) < (1 - eps) + eps / 2, u, 1 - u)
    return u


pi_det = lambda x, q: pi(x, q, 0, jax.random.PRNGKey(0))
pi_det = jax.jit(pi_det)
pi_det_single = lambda x, q: pi_single(x, q, 0, jax.random.PRNGKey(0))
pi_det_single = jax.jit(pi_det_single)

###

state1_n = 25
state1_vals = np.array(
    [
        [[x0, x1] for x1 in np.linspace(state_x1_min, state_x1_max, state1_n)]
        for x0 in np.linspace(state_x0_min, state_x0_max, state1_n)
    ]
)
state1_vals = state1_vals.reshape(-1, 2)

get_R1 = lambda r: _get_R1(r, state1_vals)
get_R1 = jax.jit(get_R1)

_get_T10 = lambda x, u: np.exp(
    _get_T0(x, u, state1_vals) - logsumexp(_get_T0(x, u, state1_vals))
)
_get_T10 = jax.vmap(_get_T10, in_axes=(None, 0))
_get_T11 = lambda x: _get_T10(x, np.arange(2))
_get_T11 = jax.jit(jax.vmap(_get_T11))
T1 = _get_T11(state1_vals)


def _interp1(x):
    i0 = np.maximum(
        0, np.minimum(1, (x[0] - state_x0_min) / (state_x0_max - state_x0_min))
    )
    i0 = np.round(i0 * (state1_n - 1)).astype(int)
    i1 = np.maximum(
        0, np.minimum(1, (x[1] - state_x1_min) / (state_x1_max - state_x1_min))
    )
    i1 = np.round(i1 * (state1_n - 1)).astype(int)
    return i0, i1


interp1_single = lambda x, q: q.reshape(state1_n, state1_n, 2)[_interp1(x)]
interp1_single = jax.jit(interp1_single)
