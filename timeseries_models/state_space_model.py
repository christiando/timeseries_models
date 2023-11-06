__author__ = "Christian Donner"
from jax import numpy as jnp
from jax import jit, lax, vmap
from timeseries_models import observation_model, state_model
from gaussian_toolbox import pdf
import pickle
import os
import numpy as onp
import time
from typing import Union, Tuple, List

##################################################################################################
# This file is part of the Gaussian Toolbox,                                                     #
#                                                                                                #
# It contains the class to fit state space models (SSMs) with the expectation-maximization       #
# algorithm.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
from jax import numpy as jnp
from jax import jit, lax

from gaussian_toolbox import pdf
import pickle
import os
import time
from typing import Union, Tuple


class StateSpaceModel:
    """Class to fit a state space model with the expectation-maximization procedure.

    :param X: Training data. Dimensions should be [T, Dx].
    :type X: jnp.ndarray
    :param observation_model: The observation model of the data.
    :type observation_model: observation_model.ObservationModel
    :param state_model: The state model for the latent variables.
    :type state_model: state_model.StateModel
    :param max_iter: Maximal number of EM iteration performed_, defaults to 100
    :type max_iter: int, optional
    :param conv_crit: Convergence criterion for the EM procedure, defaults to 1e-3
    :type conv_crit: float, optional
    :param u_x: Control variables for observation model. Leading dimensions should be [T,...], defaults to None
    :type u_x: jnp.ndarray, optional
    :param u_z:  Control variables for state model. Leading dimensions should be [T,...], defaults to None
    :type u_z: jnp.ndarray, optional
    :param timeit:  If true, prints the timings. , defaults to False
    :type timeit: bool, optional
    """

    def __init__(
        self,
        observation_model: observation_model.ObservationModel,
        state_model: state_model.StateModel,
        timeit: bool = False,
    ):
        # observation model
        self.om = observation_model
        # state model
        self.sm = state_model

    def _init(
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray = None,
        Sigma0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        horizon: int = 0,
    ) -> Tuple[jnp.ndarray]:
        assert X.shape[-1] == self.om.Dx
        assert control_x == None or control_x.ndim == X.ndim
        assert control_z == None or control_z.ndim == X.ndim
        if X.ndim == 2:
            X = X[None]
        if control_x == None:
            control_x = jnp.empty((X.shape[0], X.shape[1] + horizon, 0))
        elif control_x.ndim == 2:
            control_x = control_x[None]
        if control_z == None:
            control_z = jnp.empty((X.shape[0], X.shape[1] + horizon, 0))
        elif control_z.ndim == 2:
            control_z = control_x[None]
        assert control_x.shape[1] == (X.shape[1] + horizon)
        assert control_z.shape[1] == (X.shape[1] + horizon)
        if mu0 == None:
            mu0 = jnp.zeros((X.shape[0], 1, self.sm.Dz))
        if Sigma0 == None:
            Sigma0 = jnp.tile(jnp.eye(self.sm.Dz)[None, None], (X.shape[0], 1, 1, 1))
        return X, mu0, Sigma0, control_x, control_z

    @staticmethod
    def _check_convergence(Q_new, Q_old, conv_crit: float) -> bool:
        conv = (Q_new - Q_old) / jnp.amax(
            jnp.array([1, jnp.abs(Q_old), jnp.abs(Q_new)])
        )
        return jnp.abs(conv) < conv_crit

    def fit(
        self,
        X: jnp.ndarray,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        max_iter: int = 100,
        conv_crit: float = 1e-3,
        timeit: bool = False,
    ):
        """Fits the expectation-maximization algorithm.

        Runs until convergence or maximal number of iterations is reached.
        """
        X, mu0, Sigma0, control_x, control_z = self._init(
            X, control_x=control_x, control_z=control_z
        )
        converged = False
        iteration = 0
        Q_list = []
        Q_func_old = -jnp.inf
        while iteration < max_iter and not converged:
            time_start_total = time.perf_counter()
            smooth_dict, two_step_smooth_dict = self.estep(
                X, mu0, Sigma0, control_x, control_z
            )
            etime = time.perf_counter() - time_start_total
            time_start = time.perf_counter()
            Q_func = self.compute_Q_function(
                X, smooth_dict, two_step_smooth_dict, mu0, Sigma0, control_x, control_z
            )
            # Q_func = self.compute_predictive_log_likelihood(X[:,:-1], mu0, Sigma0, control_x, control_z)
            Q_time = time.perf_counter() - time_start
            time_start = time.perf_counter()
            mu0, Sigma0 = self.mstep(
                X, smooth_dict, two_step_smooth_dict, control_x, control_z
            )
            mtime = time.perf_counter() - time_start
            Q_list.append(Q_func)
            if iteration > 2:
                converged = self._check_convergence(Q_list[-2], Q_func, conv_crit)
            iteration += 1
            Q_func_old = Q_func
            if iteration % 1 == 0:
                print("Iteration %d - Q-function=%.1f" % (iteration, Q_func_old))
            tot_time = time.perf_counter() - time_start_total
            if timeit:
                print(
                    "###################### \n"
                    + "E-step: Run Time %.1f \n" % etime
                    + "Q-func: Run Time %.1f \n" % Q_time
                    + "M-step: Run Time %.1f \n" % mtime
                    + "Total: Run Time %.1f \n" % tot_time
                    + "###################### \n"
                )
        if not converged:
            print("EM reached the maximal number of iterations.")
        else:
            print("EM did converge.")
        p0_dict = {"Sigma": Sigma0, "mu": mu0}
        return Q_list, p0_dict, smooth_dict, two_step_smooth_dict

    def predict(
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray = None,
        Sigma0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        horizon: int = 1,
        observed_dims: jnp.ndarray = None,
        first_prediction_idx: int = 0,
        return_as_dict: bool = False
    ):
        X, mu0, Sigma0, control_x, control_z = self._init(
            X,
            mu0=mu0,
            Sigma0=Sigma0,
            control_x=control_x,
            control_z=control_z,
            horizon=horizon,
        )
        predict_func = jit(
            vmap(
                lambda X, mu0, Sigma0, control_x, control_z: self._predict(
                    X,
                    mu0,
                    Sigma0,
                    control_x,
                    control_z,
                    horizon,
                    observed_dims,
                    first_prediction_idx,
                )
            )
        )
        data_predict_dict = predict_func(X, mu0, Sigma0, control_x, control_z)
        if return_as_dict:
            return data_predict_dict
        else:
            data_prediction_densities = []
            num_batches = X.shape[0]
            for ibatch in range(num_batches):
                batch_density = pdf.GaussianPDF(
                    Sigma=data_predict_dict["Sigma"][ibatch],
                    mu=data_predict_dict["mu"][ibatch],
                    Lambda=data_predict_dict["Lambda"][ibatch],
                    ln_det_Sigma=data_predict_dict["ln_det_Sigma"][ibatch],
                )
                data_prediction_densities.append(batch_density)
            if num_batches == 1:
                return data_prediction_densities[0]
            else:
                return data_prediction_densities

    def _predict(
        self,
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
        if first_prediction_idx == 0:
            p0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
            init = p0
            prediction_step = lambda cp, vars_t: self._prediction_step(
                cp, vars_t, control_x, control_z, observed_dims, horizon
            )
            _, result = lax.scan(prediction_step, init, (X, jnp.arange(0, T)))
        else:
            filter_dict = self._forward_sweep(
                X[:first_prediction_idx],
                mu0,
                Sigma0,
                control_x[:first_prediction_idx],
                control_z[:first_prediction_idx],
            )
            p0_pred = pdf.GaussianPDF(
                Sigma=filter_dict["Sigma"][-1:],
                mu=filter_dict["mu"][-1:],
                Lambda=filter_dict["Lambda"][-1:],
                ln_det_Sigma=filter_dict["ln_det_Sigma"][-1:],
            )
            init = p0_pred
            prediction_step = lambda cp, vars_t: self._prediction_step(
                cp,
                vars_t,
                control_x[first_prediction_idx:],
                control_z[first_prediction_idx:],
                observed_dims,
                horizon,
            )
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
        return data_prediction_dict

    def _prediction_step(
        self, carry, vars_t, control_x, control_z, observed_dims, horizon
    ):
        X_t, t = vars_t
        roll_out_step = lambda cp, vars_t: self._roll_out_horizon(
            cp, vars_t, control_z, t
        )
        _, result = lax.scan(roll_out_step, carry, jnp.arange(horizon))
        (
            Sigma_prediction,
            mu_prediction,
            Lambda_prediction,
            ln_det_Sigma_prediction,
        ) = result
        cur_prediction_density = pdf.GaussianPDF(
            Sigma=Sigma_prediction[:1],
            mu=mu_prediction[:1],
            Lambda=Lambda_prediction[:1],
            ln_det_Sigma=ln_det_Sigma_prediction[:1],
        )
        cur_filter_density = self.om.partially_observed_filtering(
            cur_prediction_density, X_t[None], observed_dims, u=control_x[t]
        )
        carry = cur_filter_density
        horizon_prediction_density = pdf.GaussianPDF(
            Sigma=Sigma_prediction[-1:],
            mu=mu_prediction[-1:],
            Lambda=Lambda_prediction[-1:],
            ln_det_Sigma=ln_det_Sigma_prediction[-1:],
        )
        horizon_data_density = self.om.get_data_density(
            horizon_prediction_density, u=control_x[t + horizon]
        )
        result = (
            horizon_data_density.Sigma[0],
            horizon_data_density.mu[0],
            horizon_data_density.Lambda[0],
            horizon_data_density.ln_det_Sigma[0],
        )
        return carry, result

    def _roll_out_horizon(self, carry, vars_t, control_z, t):
        t_horizon = vars_t
        pre_prediction_density = carry
        cur_prediction_density = self.sm.prediction(
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

    def mstep(
        self, X, smooth_dict, two_step_smooth_dict, control_x, control_z
    ) -> Tuple[jnp.ndarray]:
        """Perform the maximization step, i.e. the updates of model parameters."""
        # Update parameters of state model
        self.sm.update_hyperparameters(
            smooth_dict,
            two_step_smooth_dict,
            control_z=control_z,
        )
        # Update parameters of observation model
        self.om.update_hyperparameters(X, smooth_dict, control_x=control_x)
        mu0, Sigma0 = smooth_dict["mu"][:, :1], smooth_dict["Sigma"][:, :1]
        return mu0, Sigma0

    def estep(
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray,
        Sigma0: jnp.ndarray,
        control_x: jnp.ndarray,
        control_z: jnp.ndarray,
    ) -> Tuple[dict]:
        return jit(vmap(self._estep))(X, mu0, Sigma0, control_x, control_z)

    def _estep(
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray,
        Sigma0: jnp.ndarray,
        control_x: jnp.ndarray,
        control_z: jnp.ndarray,
    ) -> Tuple[dict]:
        """Perform the expectation step, i.e. the forward-backward algorithm."""
        filter_dict = self._forward_sweep(X, mu0, Sigma0, control_x, control_z)
        smooth_dict, two_step_smooth_dict = self._backward_sweep(
            X, filter_dict, control_z
        )
        return smooth_dict, two_step_smooth_dict

    def _forward_step(self, carry: Tuple, vars_t: Tuple) -> Tuple:
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
        cur_prediction_density = self.sm.prediction(pre_filter_density, u=control_z_t)
        cur_filter_density = self.om.filtering(
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
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray,
        Sigma0: jnp.ndarray,
        control_x: jnp.ndarray,
        control_z: jnp.ndarray,
    ) -> dict:
        """Iterate forward, alternately doing prediction and filtering step."""
        pz0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
        init = pz0
        forward_step = lambda cf, vars_t: self._forward_step(cf, vars_t)
        _, result = lax.scan(
            forward_step, init, (X, control_x[:, None], control_z[:, None])
        )
        (
            Sigma_filter,
            mu_filter,
            Lambda_filter,
            ln_det_Sigma_filter,
        ) = result
        filter_dict = {
            "Sigma": jnp.concatenate([pz0.Sigma, Sigma_filter]),
            "mu": jnp.concatenate([pz0.mu, mu_filter]),
            "Lambda": jnp.concatenate([pz0.Lambda, Lambda_filter]),
            "ln_det_Sigma": jnp.concatenate([pz0.ln_det_Sigma, ln_det_Sigma_filter]),
        }
        return filter_dict

    def _backward_step(
        self, carry: Tuple, vars_t: Tuple[int, jnp.array], filter_density
    ) -> Tuple:
        """Compute one step backward in time (smoothing).

        :param carry: Observations and control variables
        :type carry: Tuple
        :param vars_t: Data for for constructing the smoothing density of the last (future) step
        :type vars_t: Tuple
        :return: Data of new smoothing density and smoothing and two step smoothing density.
        :rtype: Tuple
        """
        t, uz_t = vars_t
        cur_filter_density = filter_density.slice(jnp.array([t]))
        post_smoothing_density = carry
        cur_smoothing_density, cur_two_step_smoothing_density = self.sm.smoothing(
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
        self, X: jnp.ndarray, filter_dict: dict, control_z: jnp.ndarray
    ) -> Tuple[dict]:
        """Iterate backward doing smoothing step."""
        filter_density = pdf.GaussianPDF(**filter_dict)
        last_filter_density = filter_density.slice(jnp.array([X.shape[0]]))
        cs_init = last_filter_density
        backward_step = lambda cs, vars_t: self._backward_step(
            cs, vars_t, filter_density
        )
        t_range = jnp.arange(X.shape[0] - 1, -1, -1)
        _, result = lax.scan(backward_step, cs_init, (t_range, control_z[:, None]))
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
            Sigma=jnp.concatenate([Sigma_smooth[::-1], last_filter_density.Sigma]),
            mu=jnp.concatenate([mu_smooth[::-1], last_filter_density.mu]),
            Lambda=jnp.concatenate([Lambda_smooth[::-1], last_filter_density.Lambda]),
            ln_det_Sigma=jnp.concatenate(
                [ln_det_Sigma_smooth[::-1], last_filter_density.ln_det_Sigma]
            ),
        )
        new_two_step_smooth_density = pdf.GaussianPDF(
            Sigma=Sigma_two_step_smooth[::-1],
            mu=mu_two_step_smooth[::-1],
            Lambda=Lambda_two_step_smooth[::-1],
            ln_det_Sigma=ln_det_Sigma_two_step_smooth[::-1],
        )
        return new_smooth_density.to_dict(), new_two_step_smooth_density.to_dict()

    def compute_Q_function(
        self,
        X: jnp.ndarray,
        smooth_dict: dict,
        two_step_smooth_dict: dict,
        mu0: jnp.ndarray = None,
        Sigma0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
    ) -> float:
        X, mu0, Sigma0, control_x, control_z = self._init(
            X, mu0, Sigma0, control_x, control_z
        )
        Q_batch = jit(vmap(self._compute_Q_function_batch))(
            X, smooth_dict, two_step_smooth_dict, mu0, Sigma0, control_x, control_z
        )
        return jnp.sum(Q_batch)

    def _compute_Q_function_batch(
        self,
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
        T = X.shape[0]
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        p0 = pdf.GaussianPDF(Sigma=Sigma0, mu=mu0)
        p0_smoothing = smoothing_density.slice(jnp.array([0]))
        init_Q = p0_smoothing.integrate("log u(x)", factor=p0).squeeze()
        sm_Q = self.sm.compute_Q_function(
            smoothing_density, two_step_smoothing_density, control_z=control_z
        )
        phi = smoothing_density.slice(jnp.arange(1, T + 1))
        om_Q = self.om.compute_Q_function(phi, X, control_x=control_x)
        total_Q = init_Q + sm_Q + om_Q
        return total_Q

    def compute_predictive_log_likelihood(
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray = None,
        Sigma0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        ignore_init_samples: int = 0,
        first_prediction_idx: int = 0,
    ) -> float:
        """Compute the likelihood for given data :math:`X`.

        :param X: Data for which likelihood is computed. Dimensions should be [T, Dx].
        :type X: jnp.ndarray
        :param p0: Density for the initial latent state. If None, it is standard normal, defaults to None
        :type p0: pdf.GaussianPDF, optional
        :param u_x: Control variables for observation model. Leading dimensions should be [T,...], defaults to None
        :type u_x: jnp.ndarray, optional
        :param u_z:  Control variables for state model. Leading dimensions should be [T,...], defaults to None
        :type u_z: jnp.ndarray, optional
        :param ignore_init_samples: How many initial samples should be ignored in the beginning, defaults to 0
        :type ignore_init_samples: int, optional
        :return: Data log likelihood.
        :rtype: float
        """
        X, mu0, Sigma0, control_x, control_z = self._init(
            X, mu0, Sigma0, control_x, control_z, horizon=1
        )
        predictive_densities = self.predict(
            X,
            mu0,
            Sigma0,
            control_x,
            control_z,
            observed_dims=jnp.arange(X.shape[-1]),
            first_prediction_idx=first_prediction_idx,
        )
        llk = 0
        num_batches = X.shape[0]
        if num_batches == 1:
            llk += jnp.sum(
                predictive_densities.evaluate_ln(
                    X[0, first_prediction_idx:], element_wise=True
                )[ignore_init_samples:]
            )
        else:
            for density in predictive_densities:
                llk += jnp.sum(
                    density.evaluate_ln(X[0, first_prediction_idx:], element_wise=True)[
                        ignore_init_samples:
                    ]
                )
        return llk

    def _sample_step(
        self,
        z_old: jnp.array,
        vars_t: Tuple,
        observed_dims: jnp.ndarray,
        unobserved_dims: jnp.ndarray,
    ) -> Union[jnp.ndarray, jnp.ndarray]:
        """One time step sample for fixed observed data dimensions.

        :param z_old: Sample of latent variable in previous time step.
        :type z_old: jnp.ndarray [num_samples, Dz]
        :param rand_nums_z: Random numbers for sampling latent dimensions.
        :type rand_nums_z: jnp.ndarray [num_samples, Dz]
        :param x: Data vector for current time step.
        :type x: jnp.ndarray [1, Dx]
        :param rand_nums_x: Random numbers for sampling x.
        :type rand_nums_x: jnp.ndarray [num_samples, num_unobserved_dims]
        :param observed_dims: Observed dimensions.
        :type observed_dims: jnp.ndarray [num_observed_dims]
        :param unobserved_dims: Unobserved dimensions.
        :type unobserved_dims: jnp.ndarray [num_unobserved_dims]
        :return: Latent variable and data sample (only unobserved) for current time step.
        :rtype: Union[jnp.ndarray, jnp.ndarray] [num_samples, Dz] [num_samples, num_unobserved]
        """

        rand_nums_z_t, x_t, rand_nums_x_t, uz_t, ux_t = vars_t
        p_z = self.sm.condition_on_past(z_old, u=uz_t)
        L = jnp.linalg.cholesky(p_z.Sigma)
        z_sample = p_z.mu + jnp.einsum("abc,ac->ab", L, rand_nums_z_t)
        p_x = self.om.condition_on_z_and_observations(
            z_sample, x_t, observed_dims, unobserved_dims, ux_t=ux_t
        )
        L = jnp.linalg.cholesky(p_x.Sigma)
        x_sample = p_x.mu + jnp.einsum("abc,ac->ab", L, rand_nums_x_t)
        result = z_sample, x_sample
        return z_sample, result

    def sample_trajectory(
        self,
        X: jnp.ndarray,
        observed_dims: jnp.ndarray = None,
        p0: pdf.GaussianPDF = None,
        num_samples: int = 1,
        u_z: jnp.ndarray = None,
        u_x: jnp.ndarray = None,
    ) -> Union[jnp.ndarray, jnp.ndarray]:
        """Samples a trajectories, with fixed observed data dimensions.

        :param X: Data array containing the variabels to condition on, and indicating how long we wish to sample.
        :type X: jnp.ndarray [T, Dx]
        :param observed_dims: Dimension that are observed. If none no dimension is observed, defaults to None
        :type observed_dims: jnp.ndarray, optional [num_observed_dimensions]
        :param p0: Initial state density. If none, standard normal., defaults to None
        :type p0: pdf.GaussianPDF, optional
        :param num_samples: How many trajectories should be sampled, defaults to 1
        :type num_samples: int, optional
        :return: Samples of the latent variables, and the unobserved data dimensions.
        :rtype: Union[jnp.ndarray, jnp.ndarray] [T+1, nums_samples, Dz] [T, nums_samples, num_unobserved_dims]
        """
        T = X.shape[0]
        if u_z is None:
            u_z = jnp.empty((T, 0))
        if u_x is None:
            u_x = jnp.empty((T, 0))
        if p0 is None:
            p0 = pdf.GaussianPDF(
                Sigma=jnp.array([jnp.eye(self.Dz)]), mu=jnp.zeros((1, self.Dz))
            )
        if observed_dims is None:
            unobserved_dims = jnp.arange(self.Dx)
            num_unobserved_dims = X.shape[1]
        else:
            unobserved_dims = jnp.setxor1d(jnp.arange(self.Dx), observed_dims)
            num_unobserved_dims = len(unobserved_dims)

        init = jnp.asarray(p0.sample(num_samples)[:, 0])
        sample_step = jit(
            lambda z_old, vars_t: self._sample_step(
                z_old, vars_t, observed_dims, unobserved_dims
            )
        )
        # TODO: fix this
        rand_nums_z = objax.random.normal((T, num_samples, self.Dz))
        rand_nums_x = objax.random.normal((T, num_samples, num_unobserved_dims))

        _, result = lax.scan(
            sample_step, init, (rand_nums_z, X, rand_nums_x, u_z[:, None], u_x[:, None])
        )
        z_sample, X_sample = result

        return z_sample, X_sample
    
    def get_params(self):
        """Get the parameters of the model.

        :return: Parameters of the model.
        :rtype: dict
        """
        return {"sm_params": self.sm.get_params(), "om_params": self.om.get_params()}

    def set_params(self, params: dict):
        self.om = self.om.from_dict(params["om_params"])
        self.sm = self.sm.from_dict(params["sm_params"])
        
    def save(self, model_name: str, path: str = "", overwrite: bool = False):
        """Save the model.

        :param model_name: Name of the model, which is used as file name.
        :type model_name: str
        :param path:  Path to which model is saved to, defaults to ""
        :type path: str, optional
        :param overwrite: Overwrite existing file, defaults to False
        :type overwrite: bool, optional
        :raises RuntimeError: If file exists and overwrite is False.
        """
        if os.path.isfile(path) and not overwrite:
            raise RuntimeError(
                "File already exists. Pick another name or indicate overwrite."
            )
        else:
            params = self.get_params()
            model_dict = params | {
                "sm_class": self.sm.__class__.__name__,
                "om_class": self.om.__class__.__name__,
            }
            pickle.dump(model_dict, open(f"{path}/{model_name}.p", "wb"))

    @classmethod
    def load(cls, model_name: str, path: str = ""):
        """Load the model.

        :param model_name: Name of the model, which is used as file name.
        :type model_name: str
        :param path:  Path to which model is saved to, defaults to ""
        :type path: str, optional
        """
        model_dict = pickle.load(open("%s/%s.p" % (path, model_name), "rb"))
        state_model_dict = {
            "LinearStateModel": state_model.LinearStateModel,
            "LSEMStateModel": state_model.LSEMStateModel,
            "LRBFMStateModel": state_model.LRBFMStateModel,
            "NNControlStateModel": state_model.NNControlStateModel,
        }
        observation_model_dict = {
            "LinearObservationModel": observation_model.LinearObservationModel,
            "LSEMObservationModel": observation_model.LSEMObservationModel,
            "LRBFMObservationModel": observation_model.LRBFMObservationModel,
            "HeteroscedasticObservationModel": observation_model.HeteroscedasticObservationModel,
        }
        try:
            sm = state_model_dict[model_dict["sm_class"]].from_dict(
                model_dict["sm_params"]
            )
            om = observation_model_dict[model_dict["om_class"]].from_dict(
                model_dict["om_params"]
            )
            return cls(observation_model=om, state_model=sm)
        except KeyError:
            print(
                "Observation and/or state model not found. Please construct manually."
            )
            return model_dict
