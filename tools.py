import numpy as np
import math


__all__ = ['sumsq', 'eval_objective', 'gen_random_directions']


def sumsq(x):
    return np.dot(x, x)

# Evaluate objective function at x
def eval_objective(objfun, x):
    fval = objfun(x)
    return fval

# Generate random sample directions
def gen_random_directions(n, num_pts, delta, prand, Q=None):
    if Q is not None:
        p = Q.shape[1]
        assert Q.shape == (n, p), "Q must have n rows"
    else:
        p = 0
    assert delta > 0, "delta must be strictly positive"
    assert num_pts > 0, "num_pts must be strictly positive"
    assert num_pts <= n - p, "num_pts must be <= n-p (p=number of columns of Q)"

    results = np.zeros((num_pts, n))  # save space for results

    A = np.random.normal(size=(n, num_pts)) / math.sqrt(prand)
    if Q is not None:
        A = A - np.dot(Q, np.dot(Q.T, A))  # make orthogonal to columns of Q
    A_Q, A_R = np.linalg.qr(A, mode='reduced')  # make directions orthonormal

    # Construct transfomation matrix to make the results satisfy Haar distribution
    A_R_diag = A_R.diagonal()
    Mtrans = np.diag(A_R_diag/np.absolute(A_R_diag))
    A_Q = np.dot(A_Q, Mtrans)

    # The results are the columns of A_Q * delta
    for i in range(num_pts):
        results[i, :] = delta * A_Q[:, i]

    return results

