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

def _generate_behav(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = pi(x, q_true, 0.5, keys[0])
    x1 = p(x, u, keys[1])
    return x1, (x, u)

@jax.jit
def generate_behav(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate_behav, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_behav_batch = jax.jit(jax.vmap(generate_behav))

def _generate_opt(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = pi_det(x, q_true)
    x1 = p(x, u, keys[1])
    return x1, (x, u)

@jax.jit
def generate_opt(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate_opt, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_opt_batch = jax.jit(jax.vmap(generate_opt))

def _generate_lori(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = pi_det(x, q_lori)
    x1 = p(x, u, keys[1])
    return x1, (x, u)
    
@jax.jit
def generate_lori(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate_lori, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_lori_batch = jax.jit(jax.vmap(generate_lori))

def _generate_trex(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = pi_det_single(x, q_trex)
    x1 = p(x, u, keys[1])
    return x1, (x, u)

@jax.jit
def generate_trex(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate_trex, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_trex_batch = jax.jit(jax.vmap(generate_trex))

def _generate_birl(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = pi_det_single(x, q_birl)
    x1 = p(x, u, keys[1])
    return x1, (x, u)

@jax.jit
def generate_birl(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate_birl, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_birl_batch = jax.jit(jax.vmap(generate_birl))

@jax.jit
def nnet(params, x):
    h0 = np.tanh(params['W0'] @ x + params['b0'])
    h1 = np.tanh(params['W1'] @ h0 + params['b1'])
    y = jax.nn.softmax(params['W2'] @ h1 + params['b2'])
    return y

def _generate_bc(arg0, arg1):
    x, key = arg0, arg1
    keys = jax.random.split(key, 2)
    u = jax.random.choice(keys[0], np.arange(2, dtype='int32'), p=nnet(params, x))
    x1 = p(x, u, keys[1])
    return x1, (x, u)

@jax.jit
def generate_bc(key):
    keys = jax.random.split(key, 2)
    x0 = p0(keys[0])
    xf, (xs, us) = jax.lax.scan(_generate_bc, x0, jax.random.split(keys[1], 20))
    xs = np.concatenate((xs, xf[None,...]))
    return xs, us

generate_bc_batch = jax.jit(jax.vmap(generate_bc))

q_true = solve(.95, R_true, R_true_max, T)
with open('res/res-lori-k{}.obj'.format(args.key), 'rb') as f:
    r, r_max, *_ = dill.load(f)
    R, R_max = get_R(r), r_max
    q_lori = solve(.95, R, R_max, T)
with open('res/res-trex-k{}.obj'.format(args.key), 'rb') as f:
    r = dill.load(f)
    R = get_R(r)
    q_trex = solve_single(.95, R, T)
with open('res/res-birl-k{}.obj'.format(args.key), 'rb') as f:
    r = dill.load(f)
    R = get_R(r)
    q_birl = solve_single(.95, R, T)
with open('res/res-bc-k{}.obj'.format(args.key), 'rb') as f:
    params = dill.load(f)

subkey, key = jax.random.split(key)
traj_xs, traj_us = generate_behav_batch(jax.random.split(subkey, 1000))
with open('data/traj-behav-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((traj_xs, traj_us), f)

subkey, key = jax.random.split(key)
traj_xs, traj_us = generate_opt_batch(jax.random.split(subkey, 1000))
with open('data/traj-opt-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((traj_xs, traj_us), f)

subkey, key = jax.random.split(key)
traj_xs, traj_us = generate_lori_batch(jax.random.split(subkey, 1000))
with open('res/res-lori-traj-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((traj_xs, traj_us), f)

subkey, key = jax.random.split(key)
traj_xs, traj_us = generate_trex_batch(jax.random.split(subkey, 1000))
with open('res/res-trex-traj-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((traj_xs, traj_us), f)

subkey, key = jax.random.split(key)
traj_xs, traj_us = generate_birl_batch(jax.random.split(subkey, 1000))
with open('res/res-birl-traj-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((traj_xs, traj_us), f)

subkey, key = jax.random.split(key)
traj_xs, traj_us = generate_bc_batch(jax.random.split(subkey, 1000))
with open('res/res-bc-traj-k{}.obj'.format(args.key), 'wb') as f:
    dill.dump((traj_xs, traj_us), f)
