# QARSTA -- Quadratic Approximated Random Subspace Trust-region Algorithm
![GitHub](https://img.shields.io/badge/License-GPL%20v3-blue.svg)

## Citation
TO BE ADDED


## Installation
To install QARSTA, please download from Github:
```sh
git clone https://github.com/yiwchen233/QARSTA
```

To update to the latest version, please go to the top level directory and do:
```sh
git pull
```


## Using QARSTA
The API of QARSTA is:
```sh
sol = QARSTA.solve(obj, x0, p, prand, deltabeg, deltaend, maxfun, fmin_true, model_type, resfuns, resfun_num)
```

### Inputs
````
obj         (required)  objective function
x0          (required)  starting point
p           (required)  full subspace dimension
prand       (required)  minimum randomized subspace dimension
deltabeg    (optional)  initial trust-region radius
deltaend    (optional)  minimum trust-region radius
maxfun      (optional)  maximum number of function evaluations
model_type  (optional)  model construction technique (must be one of "quadratic", "underdetermined quadratic", "linear", or "square of linear")
resfuns     (required if model_type == "square of linear")  residue functions
resfun_num  (required if model_type == "square of linear")  number of residue functions
````

### Output
A class that contains the results of QARSTA and can be called by:
````
sol.x      minimizer obtained by QARSTA
sol.f      minimum function value obtained by QARSTA
sol.nf     number of function evaluation used
sol.niter  number of iteration used
````

### Examples
The files in the format of example_XXX.py are some examples of how to use QARSTA, where XXX corresponds to the model construction technique used in the example. 


## License 
All code in QARSTA is released under the GNU GPL license [LICENSE](/LICENSE).  


## Contact
Please contact us via email to report any issues:
```sh
yiwchen@student.ubc.ca
```