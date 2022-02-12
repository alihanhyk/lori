import argparse
import dill
import jax
import jax.numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type=int, default=0)
parser.add_argument('--silent', action='store_true')
args = parser.parse_args()

jax.config.update('jax_platform_name', 'cpu')
key = jax.random.PRNGKey(args.key)

with open('data/data-k{}.obj'.format(args.key), 'rb') as f:
    data_xs, data_us, pref_is, pref_js, *_ = dill.load(f)
    data_xs = jax.device_put(data_xs)

def _likelihood(r, i, j):
    r_i = r @ data_xs[i,:,:].mean(axis=0)
    r_j = r @ data_xs[j,:,:].mean(axis=0)
    prob = 1 / (1 + np.exp(-(r_i - r_j)))
    return np.log(prob)

_likelihood = jax.vmap(_likelihood, in_axes=(None,0,0))

@jax.jit
def likelihood(params):
    r = np.exp(params['r']) * np.array([-1,1])
    return _likelihood(r, pref_is, pref_js).sum()

grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)

params = dict()
subkey, key = jax.random.split(key)
params['r'] = .001 * jax.random.normal(subkey, shape=(2,))

grad_mnsq_params = dict()
for k in params:
    grad_mnsq_params[k] = np.zeros(params[k].shape)

likes = [None] * 10
for i in range(25000):

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

r = np.exp(params['r']) * np.array([-1,1])
if not args.silent:
    print(r)

with open('res/res-trex-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump(r, f)
