# QARSTA -- Quadratic Approximation Random Subspace Trust-region Algorithm
![GitHub](https://img.shields.io/badge/License-GPL%20v3-blue.svg)

This is the source code of the Quadratic Approximation Random Subspace Trust-region Algorithm (QARSTA) proposed in our [paper](https://link.springer.com/article/10.1007/s10589-024-00590-8).  QARSTA is a Python package originally designed for large-scale determined unconstrained optimization problems where the derivative information is unavailable.  This algorithm does not require any special structure of the objective function and is currently able to construct four types of surrogate models:
* determined quadratic interpolation model (using $\frac{(n+1)(n+2)}{2}$ sample points)  
* underdetermined quadratic interpolation model (using $2n+1$ sample points)  
* linear interpolation model (using $n+1$ sample points)  
* square of linear interpolation model (using $n+1$ sample points, can only be constructed when the objective function has the structure of sum-of-square)

For a detailed explanation, please see: Y. Chen, W. Hare, and A. Wiebe, $Q$-fully quadratic modeling and its application in a random subspace derivative-free method, Computational Optimization and Applications (2024) (https://link.springer.com/article/10.1007/s10589-024-00590-8)


## Citation
If you use our code in your research, then please cite:
```
@article{chen2024qfully,
  title = {{$Q$}-fully Quadratic Modeling and its Application in a Random Subspace Derivative-free Method}, 
  author = {Yiwen Chen and Warren Hare and Amy Wiebe},
  journal = {Computational Optimization and Applications},
  year = {2024},
  doi = {10.1007/s10589-024-00590-8},
  url = {https://link.springer.com/article/10.1007/s10589-024-00590-8}
}
```


## Requirements
QARSTA requires Python 3.11.6 or above, with the following python packages:
```
NumPy >= 1.24.2
SciPy >= 1.10.1
```


## Installation & Updating
To install QARSTA, please download from Github by either downloading the ZIP file or using the follwing command:
```sh
git clone https://github.com/yiwchen233/QARSTA
```

To update to the latest version, please go to the top-level directory and do the following:
```sh
git pull
```


## Using QARSTA
The API of QARSTA is:
```sh
sol = QARSTA.solve(obj, x0, p, prand, deltabeg, deltaend, maxfun, fmin_true, model_type, resfuns, resfun_num)
```


### Inputs
```
obj         (required)  objective function
x0          (required)  starting point
p           (required)  full subspace dimension
prand       (required)  minimum randomized subspace dimension
deltabeg    (optional, default 0.1\max(\|x0\|_\infty, 1.0))  initial trust-region radius
deltaend    (optional, default 10^{-8})  minimum trust-region radius
maxfun      (optional, default 10^5)  maximum number of function evaluations
model_type  (optional, default "quadratic")  model construction technique (must be one of "quadratic", "underdetermined quadratic", "linear", or "square of linear")
resfuns     (required if model_type == "square of linear", default None)  residue functions
resfun_num  (required if model_type == "square of linear", default None)  number of residue functions
```


### Output
A class that contains the results of QARSTA and can be called by:
```
sol.x      minimizer obtained by QARSTA
sol.f      minimum function value obtained by QARSTA
sol.nf     number of function evaluations used
sol.niter  number of iterations used
```


### Examples
The files in the format of example_XXX.py are some examples of how to use QARSTA, where XXX corresponds to the model construction technique used in the example. 


## License 
All code in QARSTA is released under the GNU GPL [license](/LICENSE).  


## Contact
Please contact us via email to report any issues:
```
yiwchen@student.ubc.ca
```
