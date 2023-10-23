import numpy as np
import QARSTA

def testfnc(x):
    return x[0]**2 + x[1]**2

def testfnc_res(x):
    return x

x0 = np.array([1.0, 1.0])

np.random.seed(0)
sol = QARSTA.solve(testfnc, x0, p = 2, prand = 1, model_type = "quadratic")
print(sol)