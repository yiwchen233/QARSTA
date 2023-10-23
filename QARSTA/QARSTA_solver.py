"""
QARSTA: Quadratic Approximated Random Subspace Trust-region Algorithm
For a given blackbox objective function f with n-dimensional variable x, 
this code solves the determined unconstrained blackbox optimization problem
min f(x).
"""

import numpy as np

from .QARSTA_exit_info import *
from .QARSTA_model import *
from .QARSTA_parameters import ParameterList
from .trust_region import trsbox
from .tools import eval_objective, gen_random_directions


__all__ = ['solve']


# Trust-region radius update procedure
def update_tr(delta, ratio, norm_sk, params):
    if ratio < params("tr_radius.eta1"):  # ratio < 0.1
        delta = params("tr_radius.gamma_dec") * delta
    elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
        delta = delta  
    else:  # ratio >= 0.7
        if norm_sk >= 0.95 * delta: 
            delta = min(params("tr_radius.gamma_inc") * delta, params("tr_radius.delta_max"))
        else:
            delta = delta 

    return delta

# Construct sample set
def construct_sample_set(model, p, delta, objfun, nf, maxfun, params, fmin_true):
    exit_info = None

    if model.p > p:
        # If we somehow have more points, remove until correct
        while model.p > p:
            k = np.argmax(model.distances_to_xiter())
            model.remove_point(k)

    if model.p < p:
        # Generate new directions orthogonal to current directions
        model.factorise_system()
        dirns = gen_random_directions(model.n, p-model.p, delta, model.prand, Q=model.Q)
        
        for i in range(dirns.shape[0]):
            d = dirns[i, :]
            xnew = model.xiter() + d

            # Evaluate objective at xnew
            nf += 1
            fnew = eval_objective(objfun, xnew)

            if fmin_true is not None and fnew <= fmin_true + params("model.rel_tol") * (model.fbeg - fmin_true):
                model.save_point(xnew, fnew)
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                break  # quit

            if nf >= maxfun:
                model.save_point(xnew, fnew) 
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                break  # quit

            # Append xnew to model
            model.append_point(xnew, fnew)

    return exit_info, model, nf


def solve_main(objfun, x0, deltabeg, deltaend, maxfun, params, p, prand, fmin_true=None, model_type="quadratic", resfuns=None, resfun_num=None):

    exit_info = None
    
    # Start with evaluating f(x0)
    nf = 1
    f0 = eval_objective(objfun, x0)
    
    if fmin_true is not None and f0 <= fmin_true + params("model.rel_tol") * (f0 - fmin_true):
        # If f(x0) is already good enough, declare success and exit
        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
        return x0, f0, nf, 0, exit_info

    # Initialize model
    delta = deltabeg
    model = SampleSet(p, prand, x0, f0, rel_tol=params("model.rel_tol"))
    exit_info, nf = model.initialise_sample_set(delta, objfun, nf, maxfun, fmin_true=fmin_true)

    if exit_info is not None:
        xiter, fiter = model.get_final_results()
        return xiter, fiter, nf, 0, exit_info

    # Start iterating
    current_iter = 0
    while True:
        current_iter += 1

        # Construct sample set
        exit_info, model, nf = construct_sample_set(model, p, delta, objfun, nf, maxfun, params, fmin_true)
        if exit_info is not None:
            break  # quit

        # Build model according to the specified model type
        if model_type == "quadratic":
            interp_ok, ck, gk, Hk, exit_info, nf = model.interpolate_quadratic_model(objfun, nf, maxfun, fmin_true)
        elif model_type == "underdetermined quadratic":
            interp_ok, ck, gk, Hk, exit_info, nf  = model.interpolate_underdetermined_quadratic_model(objfun, nf, maxfun, fmin_true)
        elif model_type == "linear":
            interp_ok, ck, gk, exit_info = model.interpolate_linear_model()
        elif model_type == "square of linear":
            interp_ok, ck, gk, Hk, exit_info = model.linear_model_square(resfuns, resfun_num)
        xk = model.xiter()
        fk = model.fiter()

        if exit_info is not None:
            break
        if not interp_ok:
            exit_info = ExitInformation(EXIT_LINALG_ERROR, "Failed to build interpolation model")
            break  # quit

        # Criticality step
        norm_gk = np.linalg.norm(gk)
        if params("general.criticality_step_mu") * norm_gk < delta:
            delta = params("tr_radius.gamma_dec") * delta
            if delta <= deltaend:
                exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached deltaend")
                break  # quit

            # Shrink all directions
            dirns = model.directions_from_xiter()
            for i in range(dirns.shape[0]):
                d = dirns[i, :]
                xnew = model.xiter() + params("tr_radius.gamma_dec") * d

                # Evaluate objective at xnew
                nf += 1
                fnew = eval_objective(objfun, xnew)

                if fmin_true is not None and fnew <= fmin_true + params("model.rel_tol") * (model.fbeg - fmin_true):
                    model.save_point(xnew, fnew)
                    exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                    break  # quit

                if nf >= maxfun:
                    model.save_point(xnew, fnew) 
                    exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                    break  # quit

                if i < model.kiter:
                    model.points[i, :] = xnew.copy()
                else:
                    model.points[i + 1, :] = xnew.copy()

            continue

        # Calculate tentative step
        if model_type == "linear":
            sk_red = -gk * delta / norm_gk
            pred_reduction = delta * norm_gk
        else:
            sk_red, _, _ = trsbox(np.zeros((model.p,)), gk, Hk, -1e20 * np.ones((model.p,)), 1e20 * np.ones((model.p,)), delta)
            pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.dot(sk_red))

        sk_full = model.Q.dot(sk_red)
        norm_sk = np.linalg.norm(sk_red)
        xnew = xk + sk_full

        # Evaluate objective at xnew
        nf += 1
        fnew = eval_objective(objfun, xnew)

        if fmin_true is not None and fnew <= fmin_true + params("model.rel_tol") * (model.fbeg - fmin_true):
            model.save_point(xnew, fnew)
            exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
            break  # quit

        if nf >= maxfun:
            model.save_point(xnew, fnew)
            exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
            break  # quit

        # Decide on type of step
        actual_reduction = fk - fnew
        ratio = actual_reduction / pred_reduction

        # Update trust region radius
        delta = update_tr(delta, ratio, norm_sk, params)

        if delta <= deltaend:
            exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached deltaend")
            break  # quit

        # Add xnew to sample set
        if model.p < model.n:
            model.append_point(xnew, fnew)
            xnew_appended = True
        else:
            # If the model is full, replace the worst point with xnew
            try:
                sigmas = model.sigmamin_corresponding_to_each_point()
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
                vals[model.kiter] = -1.0  # make sure kiter is never selected
                knew = np.argmax(vals)
            except np.linalg.LinAlgError:
                # If poisedness calculation fails, revert to dropping furthest points
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                knew = np.argmax(sqdists)
            model.change_point(knew, xnew, fnew)  # updates xiter
            xnew_appended = False

        # Update kiter (xiter)
        if model_type == "underdetermined quadratic":
            model.kiter = np.argmin(model.fvals)
        else:
            model.update_kiter()

        # Remove at least 1 direction (if xnew appended) and prand to make space for new directions
        min_npt_to_drop = model.prand + (1 if xnew_appended else 0)
        ndirs_to_keep = max(0, model.p - min_npt_to_drop)
        ndirs_to_drop = model.p - ndirs_to_keep

        # Criteria of directions to remove:
        for i in range(ndirs_to_drop):
            try:
                sigmas = model.sigmamin_corresponding_to_each_point()
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists**2 / delta**4, 1)  # BOBYQA point to remove criterion
                vals[model.kiter] = -1.0  # make sure kiter is never selected
            except np.linalg.LinAlgError:
                # If poisedness calculation fails, revert to dropping furthest points
                vals = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals[model.kiter] = -1.0  # make sure kiter is never selected
                
            k = np.argmax(vals)
            vals = np.delete(vals, k)  # keep vals indices in line with indices of model.points
            model.remove_point(k)

        # Geometry management
        if model.p > 1:
            dists = model.distances_to_xiter()
            while model.p > 1 and np.max(dists) > params("geometry.sample_set_radius_tol") * delta:
                k = np.argmax(dists)
                dists = np.delete(dists, k)
                model.remove_point(k)
        if model.p > 1:
            dirns = model.directions_from_xiter()
            current_norm = np.linalg.norm(dirns, ord=2)
            while model.p > 1 and current_norm > 1 / params("geometry.tol") and (model.points.shape)[0] > 1:
                sigmas = model.sigmamin_corresponding_to_each_point()
                sqdists = np.square(model.distances_to_xiter())  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists**2 / delta**4, 1)  # BOBYQA point to remove criterion
                vals[model.kiter] = -1.0  # make sure kiter is never selected

                k = np.argmax(vals)
                vals = np.delete(vals, k)  # keep vals indices in line with indices of model.points
                model.remove_point(k)

                dirns = model.directions_from_xiter()
                current_norm = np.linalg.norm(dirns, ord=2)

        continue

    xiter, fiter = model.get_final_results()
    
    return xiter, fiter, nf, current_iter, exit_info


def solve(objfun, x0, p, prand, deltabeg=None, deltaend=1e-8, maxfun=None, fmin_true=None, model_type="quadratic", resfuns=None, resfun_num=None):

    n = len(x0)
    assert model_type == "quadratic" or model_type == "underdetermined quadratic" or model_type == "linear" or model_type == "square of linear", "Model type must be quadratic/underdetermined quadratic/linear/square of linear"
    if model_type == "square of linear":
        assert resfuns is not None and resfun_num is not None, "For square of linear models, resfuns and resfun_num must be specified"
    assert 1 <= p <= n, "p must be in [1..n]"
    assert 1 <= prand <= p, "prand must be in [1..p]"

    if deltabeg is None:
        deltabeg = 0.1 * max(np.max(np.abs(x0)), 1.0)
    if maxfun is None:
        maxfun = 1e5

    # Set parameters
    params = ParameterList(n)

    exit_info = None
    # Input & parameter checks
    if exit_info is None and deltabeg < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "deltabeg must be strictly positive")
    if exit_info is None and deltaend < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "deltaend must be strictly positive")
    if exit_info is None and deltabeg <= deltaend:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "deltabeg must be > deltaend")
    if exit_info is None and maxfun <= 0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "maxfun must be strictly positive")
    if exit_info is None and np.shape(x0) != (n,):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be a vector")

    # Check invalid parameter values
    all_ok, bad_keys = params.check_all_params()
    if exit_info is None and not all_ok:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "Bad parameters: %s" % str(bad_keys))

    # If we had an input error, quit gracefully
    if exit_info is not None:
        exit_flag = exit_info.flag
        exit_msg = exit_info.message(with_stem=True)
        results = OptimResults(None, None, 0, 0, exit_flag, exit_msg)
        return results

    # Call main solver
    xmin, fmin, nf, niter, exit_info = solve_main(objfun, x0, deltabeg, deltaend, maxfun, params, p, prand, fmin_true=fmin_true, model_type=model_type, resfuns=resfuns, resfun_num=resfun_num)

    # Process final return values & package up
    exit_flag = exit_info.flag
    exit_msg = exit_info.message(with_stem=True)

    results = OptimResults(xmin, fmin, nf, niter, exit_flag, exit_msg)

    return results
