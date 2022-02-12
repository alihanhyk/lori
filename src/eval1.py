import dill
import jax
import jax.numpy as np
import sklearn.metrics as metrics

from simu import r_true, r_true_max, pref2

jax.config.update('jax_platform_name', 'cpu')

def _rank(x1s, x2s):
    r1 = np.minimum(r_true_max, r_true @ x1s.mean(axis=0))
    r2 = np.minimum(r_true_max, r_true @ x2s.mean(axis=0))
    return pref2(r1[0]-r2[0], r1[1]-r2[1])

_rank = jax.vmap(_rank, in_axes=(0,0))
rank = lambda traj1_xs, traj2_xs: _rank(traj1_xs, traj2_xs).mean()
rank = jax.jit(rank)

fs = ['data/traj-behav', 'res/res-bc-traj', 'res/res-birl-traj', 'res/res-trex-traj', 'res/res-lori-traj', 'data/traj-opt']
res = dict()
for f1 in fs:
    res[f1] = dict()
    for f2 in fs:
        res[f1][f2] = list()

for k in range(5):
    for f1 in fs:
        for f2 in fs:
            with open(f1 + '-k{}.obj'.format(k), 'rb') as f:
                traj1_xs, traj1_us = dill.load(f)
            with open(f2 + '-k{}.obj'.format(k), 'rb') as f:
                traj2_xs, traj2_us = dill.load(f)
            res[f1][f2].append(rank(traj1_xs, traj2_xs))


labels = ['Behavior', 'BC', 'BIRL', 'T-REX', 'LORI', 'Optimal']

print('           {}'.format(' '.join(['{:<10}               '.format(label) for label in labels])))
for f1, label1 in zip(fs, labels):
    res_mean = list()
    res_std = list()
    for f2 in fs:
        res_mean.append(str(np.array(res[f1][f2]).mean()))
        res_std.append(str(np.array(res[f1][f2]).std()))
    print('{:<8} : {}'.format(label1, ' '.join(['{:<10} ({:<12})'.format(r0, r1) for r0, r1 in zip(res_mean, res_std)])))
