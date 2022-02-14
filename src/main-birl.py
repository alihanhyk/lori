import argparse

import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

from simu import *

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", type=int, default=0)
parser.add_argument("--silent", action="store_true")
args = parser.parse_args()

jax.config.update("jax_platform_name", "cpu")
key = jax.random.PRNGKey(args.key)

with open(f"data/data-k{args.key}.obj", "rb") as f:
    data_xs, data_us, *_ = dill.load(f)


def _likelihood0(q, x, u):
    qx = interp1_single(x, q)
    return qx[u] - logsumexp(qx)


_likelihood0 = jax.vmap(_likelihood0, in_axes=(None, 0, 0))
_likelihood1 = lambda q, xs, us: _likelihood0(q, xs[:-1], us).sum()
_likelihood1 = jax.vmap(_likelihood1, in_axes=(None, 0, 0))


@jax.jit
def likelihood(_r):
    r = np.exp(_r) * np.array([-1, 1])
    q = (1 - 0.95) * solve_single(0.95, get_R1(r), T1)
    return _likelihood1(q, data_xs, data_us).sum()


def _sample(arg0, arg1):
    (_r, like), key = arg0, arg1
    keys = jax.random.split(key, 2)
    _r1 = _r + 0.01 * jax.random.normal(keys[0], shape=_r.shape)
    like1 = likelihood(_r1)
    cond = np.log(jax.random.uniform(keys[1])) < like1 - like
    _r = np.where(cond, _r1, _r)
    like = np.where(cond, like1, like)
    return (_r, like), (_r, cond)


@jax.jit
def sample(_r, like, key):
    (_r, like), (_rs, conds) = jax.lax.scan(
        _sample, (_r, like), jax.random.split(key, 1000)
    )
    return _r, like, _rs, conds.sum() / 1000


subkey, key = jax.random.split(key)
_r = 0.01 * jax.random.normal(subkey, shape=(2,))
like = likelihood(_r)

_rs_acc = np.zeros((0, 2))
for i in range(11):
    subkey, key = jax.random.split(key)
    _r, like, _rs, rate = sample(_r, like, subkey)
    _rs_acc = np.concatenate((_rs_acc, _rs))
    if not args.silent:
        print(i, rate, like)

rs = np.exp(_rs_acc) * np.array([-1, 1])[None, ...]
r = np.mean(rs[1000::100], axis=0)
if not args.silent:
    print(r)

with open(f"res/res-birl-k{args.key}.obj", "wb") as f:
    dill.dump(r, f)
