import argparse

import dill
import jax
import jax.numpy as np

from simu import pref2_long1, pref2_long1_equiv

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", type=int, default=0)
parser.add_argument("--silent", action="store_true")
args = parser.parse_args()

jax.config.update("jax_platform_name", "cpu")
key = jax.random.PRNGKey(args.key)

with open("data/liver.obj", "rb") as f:
    data = dill.load(f)

data = np.concatenate(data)
data /= data.std(axis=0)


def _likelihood(r, del_x, eps0, eps1):
    dels = np.maximum(-10, np.minimum(10, r @ del_x))
    p = pref2_long1(dels[0], dels[1], eps0, eps1)
    p_equiv = pref2_long1_equiv(dels[0], dels[1], eps0, eps1)
    return np.log(p + p_equiv / 2)


_likelihood = jax.vmap(_likelihood, in_axes=(None, 0, None, None))


@jax.jit
def likelihood(params):
    r = np.exp(params["r"])
    eps0, eps1 = np.exp(params["eps0"]), np.exp(params["eps1"])
    return _likelihood(r, data, eps0, eps1).sum()


grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)

params = dict()
subkey, key = jax.random.split(key)
params["r"] = 0.001 * jax.random.normal(subkey, shape=(2, 2))
subkey, key = jax.random.split(key)
params["eps0"] = 0.001 * jax.random.normal(subkey)
subkey, key = jax.random.split(key)
params["eps1"] = 0.001 * jax.random.normal(subkey)

grad_mnsq_params = dict()
for k in params:
    grad_mnsq_params[k] = np.zeros(params[k].shape)

likes = [None] * 10
for i in range(25000):

    grad_params = grad_likelihood(params)
    for k in params:
        grad_mnsq_params[k] = 0.9 * grad_mnsq_params[k] + 0.1 * grad_params[k] ** 2
        params[k] = params[k] + 0.001 * grad_params[k] / np.sqrt(
            grad_mnsq_params[k] + 1e-6
        )

    likes[1:] = likes[:-1]
    likes[0] = likelihood(params)
    if likes[-1] is not None and likes[0] - likes[-1] < 1e-6:
        break

    if not args.silent:
        print(i, likes[0])

r = np.exp(params["r"])
eps0, eps1 = np.exp(params["eps0"]), np.exp(params["eps1"])

if not args.silent:
    print(r)
    print(eps0, eps1)

with open("res/res-lori-liver.obj", "wb") as f:
    dill.dump((r, eps0, eps1), f)
