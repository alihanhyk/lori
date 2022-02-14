import argparse

import dill
import jax
import jax.numpy as np

from simu import pref2_long, softmin

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", type=int, default=0)
parser.add_argument("--silent", action="store_true")
args = parser.parse_args()

jax.config.update("jax_platform_name", "cpu")
key = jax.random.PRNGKey(args.key)

with open(f"data/data-k{args.key}.obj", "rb") as f:
    data_xs, data_us, pref_is, pref_js, *_ = dill.load(f)
    data_xs = jax.device_put(data_xs)


def _likelihood(r, r_max, i, j, eps0, eps1):
    r_i = softmin(10 * r_max, 10 * r @ data_xs[i, :, :].mean(axis=0)) / 10
    r_j = softmin(10 * r_max, 10 * r @ data_xs[j, :, :].mean(axis=0)) / 10
    del0 = np.maximum(-10, np.minimum(10, r_i[0] - r_j[0]))
    del1 = np.maximum(-10, np.minimum(10, r_i[1] - r_j[1]))
    return np.log(pref2_long(del0, del1, eps0, eps1))


_likelihood = jax.vmap(_likelihood, in_axes=(None, None, 0, 0, None, None))


@jax.jit
def likelihood(params):
    r = np.exp(params["r"]) * np.array([[-1, 1], [-1, 1]])
    r_max = np.array([np.exp(params["r_max"]), np.inf])
    eps0, eps1 = np.exp(params["eps0"]), np.exp(params["eps1"])
    return _likelihood(r, r_max, pref_is, pref_js, eps0, eps1).sum()


grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)

params = dict()
subkey, key = jax.random.split(key)
params["r"] = 0.001 * jax.random.normal(subkey, shape=(2, 2))
subkey, key = jax.random.split(key)
params["r_max"] = 0.001 * jax.random.normal(subkey)
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
            1e-6 + grad_mnsq_params[k]
        )

    likes[1:] = likes[:-1]
    likes[0] = likelihood(params)
    if likes[-1] is not None and likes[0] - likes[-1] < 1e-6:
        break

    if not args.silent:
        print(i, likes[0])

r = np.exp(params["r"]) * np.array([[-1, 1], [-1, 1]])
r_max = np.array([np.exp(params["r_max"]), np.inf])
eps0, eps1 = np.exp(params["eps0"]), np.exp(params["eps1"])
if not args.silent:
    print(r)
    print(r_max)

with open(f"res/res-lori-k{args.key}.obj", "wb") as f:
    dill.dump((r, r_max, eps0, eps1), f)
