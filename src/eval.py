import dill
import jax
import jax.numpy as np

from simu import r_true, r_true_max, pref2, pref2_long, softmin

jax.config.update('jax_platform_name', 'cpu')

def true(data_xs, i, j):
    r_i = np.minimum(r_true_max, r_true @ data_xs[i,:,:].mean(axis=0))
    r_j = np.minimum(r_true_max, r_true @ data_xs[j,:,:].mean(axis=0))
    return pref2(r_i[0]-r_j[0], r_i[1]-r_j[1])

true = jax.vmap(true, in_axes=(None,0,0))
true = jax.jit(true)

def predict(r, r_max, data_xs, i, j, eps0, eps1):
    r_i = softmin(10 * r_max, 10 * r @ data_xs[i,:,:].mean(axis=0)) / 10
    r_j = softmin(10 * r_max, 10 * r @ data_xs[j,:,:].mean(axis=0)) / 10
    return pref2_long(r_i[0]-r_j[0], r_i[1]-r_j[1], eps0, eps1)

predict = jax.vmap(predict, in_axes=(None,None,None,0,0,None,None))
predict = jax.jit(predict)

def predict_single(r, data_xs, i, j):
    r_i = r @ data_xs[i,:,:].mean(axis=0)
    r_j = r @ data_xs[j,:,:].mean(axis=0)
    prob = 1 / (1 + np.exp(-(r_i - r_j)))
    return prob

predict_single = jax.vmap(predict_single, in_axes=(None,None,0,0))
predict_single = jax.jit(predict_single)

err = list()
acc = list()
err_trex = list()
acc_trex = list()
err_birl = list()
acc_birl = list()

for k in range(5):

    with open('data/data-k{}.obj'.format(k), 'rb') as f:
        data_xs, data_us, _, _, pref_test_is, pref_test_js = dill.load(f)
        data_xs = jax.device_put(data_xs)

    y = true(data_xs, pref_test_is, pref_test_js)

    with open('res/res-lori-k{}.obj'.format(k), 'rb') as f:
        r, r_max, eps0, eps1 = dill.load(f)
        y_hat = predict(r, r_max, data_xs, pref_test_is, pref_test_js, eps0, eps1)

    with open('res/res-trex-k{}.obj'.format(k), 'rb') as f:
        r_trex = dill.load(f)
        y_hat_trex = predict_single(r_trex, data_xs, pref_test_is, pref_test_js)

    with open('res/res-birl-k{}.obj'.format(k), 'rb') as f:
        r_birl = dill.load(f)
        y_hat_birl = predict_single(r_birl, data_xs, pref_test_is, pref_test_js)

    err.append(np.mean((y-y_hat)**2)**.5)
    acc.append(np.mean(y_hat > 0.5))
    err_trex.append(np.mean((y-y_hat_trex)**2)**.5)
    acc_trex.append(np.mean(y_hat_trex > 0.5))
    err_birl.append(np.mean((y-y_hat_birl)**2)**.5)
    acc_birl.append(np.mean(y_hat_birl > 0.5))

err = np.array(err)
acc = np.array(acc)
err_trex = np.array(err_trex)
acc_trex = np.array(acc_trex)
err_birl = np.array(err_birl)
acc_birl = np.array(acc_birl)

print('BIRL')
print(f'    RMSE: {err_birl.mean()} ({err_birl.std()})')
print(f'    Accuracy: {acc_birl.mean()} ({acc_birl.std()})')

print('T-REX')
print(f'    RMSE: {err_trex.mean()} ({err_trex.std()})')
print(f'    Accuracy: {acc_trex.mean()} ({acc_trex.std()})')

print('LORI')
print(f'    RMSE: {err.mean()} ({err.std()})')
print(f'    Accuracy: {acc.mean()} ({acc.std()})')
