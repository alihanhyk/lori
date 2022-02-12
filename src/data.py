import argparse
import dill
import jax
import jax.numpy as np

from simu import *

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type=int, default=0)
args = parser.parse_args()

jax.config.update('jax_platform_name', 'cpu')
key = jax.random.PRNGKey(args.key)

q_opt = solve(.95, R_true, R_true_max, T)
def _generate(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = pi(x, q_opt, 0.5, keys[0])
    x1 = p(x, u, keys[1])
    return x1, (x, u)

@ jax.jit
def generate(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_batch = jax.vmap(generate)
generate_batch = jax.jit(generate_batch)

subkey, key = jax.random.split(key)
data_xs, data_us = generate_batch(jax.random.split(subkey, 1000))

@jax.jit
def rank(data_xs, data_us, key):
    keys = jax.random.split(key, 3)
    i = jax.random.randint(keys[0], (1,), minval=0, maxval=data_xs.shape[0])[0]
    j = jax.random.randint(keys[1], (1,), minval=0, maxval=i)[0]
    r_i = np.minimum(r_true_max, r_true @ data_xs[i,:,:].mean(axis=0))
    r_j = np.minimum(r_true_max, r_true @ data_xs[j,:,:].mean(axis=0))
    prob = pref2(r_i[0]-r_j[0], r_i[1]-r_j[1])
    unif = jax.random.uniform(keys[2])
    i1 = np.where(unif < prob, i, j)
    j1 = np.where(unif < prob, j, i)
    return i1, j1

rank_batch = jax.vmap(rank, in_axes=(None,None,0))
rank_batch = jax.jit(rank_batch)

subkey, key = jax.random.split(key)
pref_is, pref_js = rank_batch(data_xs, data_us, jax.random.split(subkey, 1000))

subkey, key = jax.random.split(key)
pref_test_is, pref_test_js = rank_batch(data_xs, data_us, jax.random.split(subkey, 1000))

with open('data/data-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((data_xs, data_us, pref_is, pref_js, pref_test_is, pref_test_js), f)
