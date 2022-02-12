import dill
import numpy as np

with open('data/liver.obj', 'rb') as f:
    data = dill.load(f)
    data = np.concatenate(data)

with open('res/res-lori-liver.obj', 'rb') as f:
    r, eps0, eps1 = dill.load(f)

with open('res/res-trex-liver.obj', 'rb') as f:
    r_trex = dill.load(f)

r /= data.std(axis=0)[None,...]
r_trex /= data.std(axis=0)

print('T-REX:')
print(f'    r = {r_trex}')

print('LORI:')
print(f'    r0 = {r[0]}')
print(f'    r1 = {r[1]}')
print(f'    eps0 = {eps0}')
print(f'    eps1 = {eps1}')
