import argparse
import dill
import jax
import jax.numpy as np

from simu import *

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type=int, default=0)
parser.add_argument('--silent', action='store_true')
args = parser.parse_args()

jax.config.update('jax_platform_name', 'cpu')
key = jax.random.PRNGKey(args.key)

with open('data/data-k{}.obj'.format(args.key), 'rb') as f:
    data_xs, data_us, *_ = dill.load(f)

@jax.jit
def nnet(params, x):
    h0 = np.tanh(params['W0'] @ x + params['b0'])
    h1 = np.tanh(params['W1'] @ h0 + params['b1'])
    y = jax.nn.softmax(params['W2'] @ h1 + params['b2'])
    return y

def _likelihood0(params, x, u):
    y = nnet(params, x)
    return np.log(y[u])

_likelihood0 = jax.vmap(_likelihood0, in_axes=(None,0,0))
_likelihood1 = lambda params, xs, us: _likelihood0(params, xs[:-1], us).sum()
_likelihood1 = jax.vmap(_likelihood1, in_axes=(None,0,0))
likelihood = lambda params: _likelihood1(params, data_xs, data_us).sum()
likelihood = jax.jit(likelihood)

grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)

params = dict()
subkey, key = jax.random.split(key)
subkeys = jax.random.split(subkey, 6)
params['W0'] = .001 * jax.random.normal(subkeys[0], shape=(64,2))
params['b0'] = .001 * jax.random.normal(subkeys[1], shape=(64,))
params['W1'] = .001 * jax.random.normal(subkeys[2], shape=(64,64))
params['b1'] = .001 * jax.random.normal(subkeys[3], shape=(64,))
params['W2'] = .001 * jax.random.normal(subkeys[4], shape=(2,64))
params['b2'] = .001 * jax.random.normal(subkeys[5], shape=(2,))

grad_mnsq_params = dict()
for k in params:
    grad_mnsq_params[k] = np.zeros(params[k].shape)

likes = [None] * 100
for i in range(10000):

    grad_params = grad_likelihood(params)
    for k in params:
        grad_mnsq_params[k] = .9 * grad_mnsq_params[k] + .1 * grad_params[k]**2
        params[k] = params[k] + .001 * grad_params[k] / np.sqrt(1e-6+grad_mnsq_params[k])

    likes[1:] = likes[:-1]
    likes[0] = likelihood(params)
    if likes[-1] is not None and likes[0]-likes[-1] < 1e-6:
        break

    if not args.silent:
        print(i, likes[0])

with open('res/res-bc-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump(params, f)
