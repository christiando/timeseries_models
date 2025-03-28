from jax import numpy as jnp
from jax import lax
from typing import Tuple
from gaussian_toolbox import pdf
from functools import partial
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
@partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0))
def _compute_Q_function_batch(
    params: dict,
    _init_func: callable,
    X: jnp.ndarray,
    smooth_dict: dict,
    two_step_smooth_dict: dict,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    control_x: jnp.ndarray,
    control_z: jnp.ndarray,
) -> float:
    r"""Compute Q-function.

    .. math::

        Q(w,w_{\rm old}) = \mathbb{E}\left[\ln p(Z_0\vert w)\right] + \sum_{t=1}^T\mathbb{E}\left[\ln p(X_t\vert Z_t, w)\right] + \sum_{t=1}^T\mathbb{E}\left[\ln p(Z_t\vert Z_{t-1}, w)\right],

    where the expectation is over the smoothing density :math:`q(Z_{0:T}\vert w_{\rm old})`.

    :return: Evluated Q-function.
    :rtype: float
    """
    sm, om = _init_func(params)
    smoothing_density = pdf.GaussianPDF(**smooth_dict)
    two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
    p0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
    p0_smoothing = smoothing_density.slice(jnp.array([0]))
    init_Q = p0_smoothing.integrate("log u(x)", factor=p0).squeeze()
    sm_Q = sm.compute_Q_function(
        smoothing_density, two_step_smoothing_density, control_z=control_z
    )
    #phi = smoothing_density.slice(jnp.arange(0, T))
    om_Q = om.compute_Q_function(smoothing_density, X, control_x=control_x)
    total_Q = init_Q + sm_Q + om_Q
    return total_Q

@partial(jit, static_argnums=(1,))
def _mstep(params: dict,
    _init_func: callable, 
    X, smooth_dict, 
    two_step_smooth_dict, 
    control_x, 
    control_z
    ) -> Tuple[jnp.ndarray]:
    sm, om = _init_func(params)
    sm.update_hyperparameters(
        smooth_dict,
        two_step_smooth_dict,
        control_z=control_z,
        )
    # Update parameters of observation model
    om.update_hyperparameters(X, smooth_dict, control_x=control_x)
    return {'sm_params': sm.get_params(), 'om_params': om.get_params()}

@partial(jit, static_argnums=(1,))
@partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
def _estep(
    params: dict,
    _init_func: callable,
    X: jnp.ndarray,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    control_x: jnp.ndarray,
    control_z: jnp.ndarray,
) -> Tuple[dict]:
    """Perform the expectation step, i.e. the forward-backward algorithm."""
    sm, om = _init_func(params)
    filter_dict = _forward_sweep(om, sm, X, mu0, Sigma0, control_x, control_z)
    smooth_dict, two_step_smooth_dict = _backward_sweep(
        sm, X, filter_dict, control_z
    )
    return smooth_dict, two_step_smooth_dict

def _forward_step(om, sm, carry: Tuple, vars_t: Tuple) -> Tuple:
    """Compute one step forward in time (prediction & filter).

    :param carry: Observations and control variables
    :type carry: Tuple
    :param vars_t: Data for for constructing the filter density of the last step
    :type vars_t: Tuple
    :return: Data of new filter density and prediction and filter density.
    :rtype: Tuple
    """
    X_t, control_x_t, control_z_t = vars_t
    pre_filter_density = carry
    cur_prediction_density = sm.prediction(pre_filter_density, u=control_z_t)
    cur_filter_density = om.filtering(
        cur_prediction_density, X_t[None], u=control_x_t
    )
    carry = cur_filter_density
    result = (
        cur_filter_density.Sigma[0],
        cur_filter_density.mu[0],
        cur_filter_density.Lambda[0],
        cur_filter_density.ln_det_Sigma[0],
    )
    return carry, result

def _forward_sweep(
    om,
    sm,
    X: jnp.ndarray,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    control_x: jnp.ndarray,
    control_z: jnp.ndarray,
) -> dict:
    """Iterate forward, alternately doing prediction and filtering step."""
    pz0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
    init_density = om.filtering(
        pz0, X[:1], u=control_x[:1]
    )
    #init = pz0
    def forward_step(carry, vars_t):
        return _forward_step(om, sm, carry, vars_t)
    
    _, result = lax.scan(
        forward_step, init_density, (X[1:], control_x[1:, None], control_z[:-1, None])
    )
    (
        Sigma_filter,
        mu_filter,
        Lambda_filter,
        ln_det_Sigma_filter,
    ) = result
    filter_dict = {
        "Sigma": jnp.concatenate([init_density.Sigma, Sigma_filter]),
        "mu": jnp.concatenate([init_density.mu, mu_filter]),
        "Lambda": jnp.concatenate([init_density.Lambda, Lambda_filter]),
        "ln_det_Sigma": jnp.concatenate([pz0.ln_det_Sigma, ln_det_Sigma_filter]),
    }
    return filter_dict

def _backward_step(
    sm, carry: Tuple, vars_t: Tuple[int, jnp.array]
) -> Tuple:
    """Compute one step backward in time (smoothing).

    :param carry: Observations and control variables
    :type carry: Tuple
    :param vars_t: Data for for constructing the smoothing density of the last (future) step
    :type vars_t: Tuple
    :return: Data of new smoothing density and smoothing and two step smoothing density.
    :rtype: Tuple
    """
    t, uz_t, mu_f, Sigma_f, Lambda_f, ln_det_Sigma_f = vars_t
    #filter_dict = {k: jnp.array(v) for k, v in filter_dict.items()}
    cur_filter_density = pdf.GaussianPDF(Sigma=Sigma_f, mu=mu_f, Lambda=Lambda_f, ln_det_Sigma=ln_det_Sigma_f)
    post_smoothing_density = carry
    cur_smoothing_density, cur_two_step_smoothing_density = sm.smoothing(
        cur_filter_density, post_smoothing_density, u=uz_t
    )
    carry = cur_smoothing_density
    result = (
        cur_smoothing_density.Sigma[0],
        cur_smoothing_density.mu[0],
        cur_smoothing_density.Lambda[0],
        cur_smoothing_density.ln_det_Sigma[0],
        cur_two_step_smoothing_density.Sigma[0],
        cur_two_step_smoothing_density.mu[0],
        cur_two_step_smoothing_density.Lambda[0],
        cur_two_step_smoothing_density.ln_det_Sigma[0],
    )
    return carry, result

def _backward_sweep(
    sm, X: jnp.ndarray, filter_dict: dict, control_z: jnp.ndarray
) -> Tuple[dict]:
    """Iterate backward doing smoothing step."""
    filter_density = pdf.GaussianPDF(**filter_dict)
    last_filter_density = filter_density.slice(jnp.array([-1]))
    cs_init = last_filter_density
    
    def backward_step(carry, vars_t):
        return _backward_step(sm, carry, vars_t)
    t_range = jnp.arange(0, X.shape[0]-1)
    _, result = lax.scan(backward_step, cs_init, (t_range, control_z[:-1, None], filter_density.mu[:-1,None], 
                                                    filter_density.Sigma[:-1,None], filter_density.Lambda[:-1,None], filter_density.ln_det_Sigma[:-1,None]), reverse=True)
    (
        Sigma_smooth,
        mu_smooth,
        Lambda_smooth,
        ln_det_Sigma_smooth,
        Sigma_two_step_smooth,
        mu_two_step_smooth,
        Lambda_two_step_smooth,
        ln_det_Sigma_two_step_smooth,
    ) = result
    new_smooth_density = pdf.GaussianPDF(
        Sigma=jnp.concatenate([Sigma_smooth[:], last_filter_density.Sigma]),
        mu=jnp.concatenate([mu_smooth[:], last_filter_density.mu]),
        Lambda=jnp.concatenate([Lambda_smooth[:], last_filter_density.Lambda]),
        ln_det_Sigma=jnp.concatenate([ln_det_Sigma_smooth[:], last_filter_density.ln_det_Sigma])
    )
    new_two_step_smooth_density = pdf.GaussianPDF(
        Sigma=Sigma_two_step_smooth,
        mu=mu_two_step_smooth,
        Lambda=Lambda_two_step_smooth,
        ln_det_Sigma=ln_det_Sigma_two_step_smooth,
    )
    return new_smooth_density.to_dict(), new_two_step_smooth_density.to_dict()

@partial(jit, static_argnums=(1,7,9))
@partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0, None, None, None))
def _predict(
    params: dict,
    _init_func: callable,
    X: jnp.ndarray,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    control_x: jnp.ndarray,
    control_z: jnp.ndarray,
    horizon: int,
    observed_dims: jnp.ndarray,
    first_prediction_idx: int,
):
    T = X.shape[0]
    sm, om = _init_func(params)
    if first_prediction_idx == 0:
        p0_pred = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
        init = p0_pred
        
        def prediction_step(carry, vars_t):
            return _prediction_step(om, sm, carry, vars_t, control_x, control_z, observed_dims, horizon)
        _, result = lax.scan(prediction_step, init, (X, jnp.arange(0, T)))
    else:
        filter_dict = _forward_sweep(
            om,
            sm,
            X[:first_prediction_idx],
            mu0,
            Sigma0,
            control_x[:first_prediction_idx],
            control_z[:first_prediction_idx],
        )
        last_filter_density = pdf.GaussianPDF(Sigma=filter_dict["Sigma"][-1:], 
                                                mu=filter_dict["mu"][-1:], 
                                                Lambda=filter_dict["Lambda"][-1:], 
                                                ln_det_Sigma=filter_dict["ln_det_Sigma"][-1:])
        p0_pred = sm.prediction(last_filter_density, u=control_z[first_prediction_idx])
        #p0_pred = pdf.GaussianPDF(
        #    Sigma=filter_dict["Sigma"][:],
        #    mu=filter_dict["mu"][:],
        #    Lambda=filter_dict["Lambda"][:],
        #    ln_det_Sigma=filter_dict["ln_det_Sigma"][:],
        #)
        init = p0_pred
        def prediction_step(carry, vars_t):
            return _prediction_step(om, sm, carry, vars_t, control_x, control_z, observed_dims, horizon)
        _, result = lax.scan(
            prediction_step,
            init,
            (X[first_prediction_idx:], jnp.arange(0, T - first_prediction_idx)),
        )
    data_prediction_dict = {
        "Sigma": result[0],
        "mu": result[1],
        "Lambda": result[2],
        "ln_det_Sigma": result[3],
    }
    latent_prediction_dict = {
        "Sigma": result[4],
        "mu": result[5],
        "Lambda": result[6],
        "ln_det_Sigma": result[7],
    }
    return data_prediction_dict, latent_prediction_dict

def _prediction_step(
    om, sm, carry, vars_t, control_x, control_z, observed_dims, horizon
):
    X_t, t = vars_t
    
    def roll_out_step(cp, vars_t):
        return _roll_out_horizon(sm, cp, vars_t, control_z, t)

    cur_prediction_density = carry
    cur_filter_density = om.partially_observed_filtering(
        cur_prediction_density, X_t[None], observed_dims, u=control_x[t]
    )
    next_prediction_density =sm.prediction(cur_filter_density, u=control_z[t])   
    if horizon > 1:
        _, result = lax.scan(roll_out_step, carry, jnp.arange(horizon-1))
        (
            Sigma_prediction,
            mu_prediction,
            Lambda_prediction,
            ln_det_Sigma_prediction,
        ) = result
        horizon_prediction_density = pdf.GaussianPDF(
            Sigma=Sigma_prediction[-1:],
            mu=mu_prediction[-1:],
            Lambda=Lambda_prediction[-1:],
            ln_det_Sigma=ln_det_Sigma_prediction[-1:],
        )
    else:
        horizon_prediction_density = cur_prediction_density

    carry = next_prediction_density
    
    horizon_data_density = om.get_data_density(
        horizon_prediction_density, u=control_x[t + horizon - 1]
    )
    result = (
        horizon_data_density.Sigma[0],
        horizon_data_density.mu[0],
        horizon_data_density.Lambda[0],
        horizon_data_density.ln_det_Sigma[0],
        horizon_prediction_density.Sigma[0],
        horizon_prediction_density.mu[0],
        horizon_prediction_density.Lambda[0],
        horizon_prediction_density.ln_det_Sigma[0],
    )
    return carry, result

def _roll_out_horizon(sm, carry, vars_t, control_z, t):
    t_horizon = vars_t
    pre_prediction_density = carry
    cur_prediction_density = sm.prediction(
        pre_prediction_density, u=control_z[t + t_horizon][None]
    )
    carry = cur_prediction_density
    result = (
        cur_prediction_density.Sigma[0],
        cur_prediction_density.mu[0],
        cur_prediction_density.Lambda[0],
        cur_prediction_density.ln_det_Sigma[0],
    )
    return carry, result