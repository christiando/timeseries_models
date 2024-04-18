__author__ = "Christian Donner"
from jax import numpy as jnp
from jax import scipy as jsc
from jax import jit, lax, vmap
from timeseries_models.hidden_markov_models import observation_model, state_model
import pickle
import os
import time
from typing import Tuple

##################################################################################################
# This file is part of the Gaussian Toolbox,                                                     #
#                                                                                                #
# It contains the class to fit state space models (SSMs) with the expectation-maximization       #
# algorithm.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################


class HiddenMarkovModel:
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
        observation_model: ...,
        state_model: ...,
        timeit: bool = False,
    ):
        # observation model
        self.om = observation_model
        # state model
        self.sm = state_model

    def _init(
        self,
        X: jnp.ndarray,
        pi0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        horizon: int = 1,
    ) -> Tuple[jnp.ndarray]:
        assert X.shape[-1] == self.om.Dx
        assert control_x == None or control_x.ndim == X.ndim
        assert control_z == None or control_z.ndim == X.ndim
        if X.ndim == 2:
            X = X[None]
        if control_x == None:
            control_x = jnp.empty((X.shape[0], X.shape[1] + horizon - 1, 0))
        elif control_x.ndim == 2:
            control_x = control_x[None]
        if control_z == None:
            control_z = jnp.empty((X.shape[0], X.shape[1] + horizon - 1, 0))
        elif control_z.ndim == 2:
            control_z = control_x[None]
        assert control_x.shape[1] == (X.shape[1] + horizon - 1)
        assert control_z.shape[1] == (X.shape[1] + horizon - 1)
        if pi0 == None:
            pi0 = jnp.ones((X.shape[0], 1, self.sm.Dz)) / self.sm.Dz
        return X, pi0, control_x, control_z
    
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
        X, pi0, control_x, control_z = self._init(
            X, control_x=control_x, control_z=control_z
        )
        converged = False
        iteration = 0
        llk_list = []
        llk_old = -jnp.inf
        while iteration < max_iter and not converged:
            time_start_total = time.perf_counter()
            pi_marg, pi_two_step, ln_marginal_llk = self.estep(
                X, pi0, control_x, control_z
            )
            etime = time.perf_counter() - time_start_total
            
            # Q_func = self.compute_predictive_log_likelihood(X[:,:-1], mu0, Sigma0, control_x, control_z)
            
            time_start = time.perf_counter()
            pi0 = self.mstep(
                X, pi_marg, pi_two_step, control_x, control_z
            )
            mtime = time.perf_counter() - time_start
            time_start = time.perf_counter()
            llk = jnp.sum(ln_marginal_llk)
            # self.compute_Q_function(
            #     X, smooth_dict, two_step_smooth_dict, mu0, Sigma0, control_x, control_z
            # )
            llk_time = time.perf_counter() - time_start
            llk_list.append(llk)
            if iteration > 2:
                converged = self._check_convergence(llk_list[-2], llk, conv_crit)
            iteration += 1
            llk_old = llk
            if iteration % 1 == 0:
                print("Iteration %d - Log likelihood=%.1f" % (iteration, llk_old))
            tot_time = time.perf_counter() - time_start_total
            if timeit:
                print(
                    "###################### \n"
                    + "E-step: Run Time %.1f \n" % etime
                    + "LLK-func: Run Time %.1f \n" % llk_time
                    + "M-step: Run Time %.1f \n" % mtime
                    + "Total: Run Time %.1f \n" % tot_time
                    + "###################### \n"
                )
        if not converged:
            print("EM reached the maximal number of iterations.")
        else:
            print("EM did converge.")
        p0_dict = {"pi": pi0}
        return llk_list, p0_dict, pi_marg, pi_two_step

    def mstep(
        self, X, pi_marg, pi_two_step, control_x, control_z
    ) -> Tuple[jnp.ndarray]:
        """Perform the maximization step, i.e. the updates of model parameters."""
        # Update parameters of state model
        self.sm.update_hyperparameters(
            pi_marg,
            pi_two_step,
            control_z=control_z,
        )
        # Update parameters of observation model
        self.om.update_hyperparameters(X, pi_marg, control_x=control_x)
        pi0 = pi_marg["pi"][:, :1]
        return pi0

    def estep(
        self,
        X: jnp.ndarray,
        pi0: jnp.ndarray,
        control_x: jnp.ndarray,
        control_z: jnp.ndarray,
    ) -> Tuple[dict]:
        return jit(vmap(self._estep))(X, pi0, control_x, control_z)

    def _estep(
        self,
        X: jnp.ndarray,
        pi0: jnp.ndarray,
        control_x: jnp.ndarray,
        control_z: jnp.ndarray,
    ) -> Tuple[dict]:
        """Perform the expectation step, i.e. the forward-backward algorithm."""
        ln_pX = self.om.get_log_likelihoods(X, control_x=control_x) # [T x Dz]
        fwd_messages, ln_marginal_likelihood = self._forward_sweep(ln_pX, pi0, control_z)
        bwd_messages = self._backward_sweep(
            ln_pX, control_z
        )
        smooth_dict, two_step_smooth_dict = self.compute_marginals(
            ln_pX, fwd_messages, bwd_messages, control_z
        )
        return smooth_dict, two_step_smooth_dict, ln_marginal_likelihood
    
    def compute_marginals(
        self,
        ln_pX: jnp.ndarray,
        fwd_messages: dict,
        bwd_messages: dict,
        control_z: jnp.ndarray,
    ) -> Tuple[dict]:
        """Compute the marginals of the latent variables."""
        ln_pi = jnp.log(fwd_messages) + jnp.log(bwd_messages)
        ln_pi -= ln_pX
        ln_norm = jsc.special.logsumexp(ln_pi, axis=-1)
        pi = jnp.exp(ln_pi - ln_norm[..., None])    
        ln_pi_two_step = jnp.log(fwd_messages[:-1,:,None]) + jnp.log(bwd_messages[1:,None])
        transition_mats = self.sm.get_transition_matrices(control_z=control_z)
        ln_pi_two_step += jnp.log(jnp.swapaxes(transition_mats, axis1=-1, axis2=-2))
        ln_norm_two_step = jsc.special.logsumexp(jsc.special.logsumexp(ln_pi_two_step, axis=-1), axis=-1)
        pi_two_step = jnp.exp(ln_pi_two_step - ln_norm_two_step[..., None, None])
        return {"pi": pi}, {"pi": pi_two_step}
    
    def _forward_step(self, carry: Tuple, vars_t: Tuple) -> Tuple:
        """Compute one step forward in time (prediction & filter).

        :param carry: Observations and control variables
        :type carry: Tuple
        :param vars_t: Data for for constructing the filter density of the last step
        :type vars_t: Tuple
        :return: Data of new filter density and prediction and filter density.
        :rtype: Tuple
        """
        ln_pX_t, control_z_t = vars_t
        pre_fwd_message = carry
        cur_prediction = self.sm.prediction(pre_fwd_message, u=control_z_t)
        ln_cur_fwd_message = jnp.log(cur_prediction) + ln_pX_t
        ln_marginal_llk_t = jsc.special.logsumexp(ln_cur_fwd_message, axis=-1)
        cur_fwd_message = jnp.exp(ln_cur_fwd_message - ln_marginal_llk_t[..., None])
        carry = cur_fwd_message
        result = (
            cur_fwd_message, ln_marginal_llk_t
        )
        return carry, result

    def _forward_sweep(
        self,
        ln_pX: jnp.ndarray,
        pi0: jnp.ndarray,
        control_z: jnp.ndarray,
    ) -> dict:
        """Iterate forward, alternately doing prediction and filtering step."""
        
        ln_init_density = jnp.log(pi0) + ln_pX[:1]
        ln_margin_llk_0 = jsc.special.logsumexp(ln_init_density, axis=-1)
        init_density = jnp.exp(ln_init_density - ln_margin_llk_0[..., None])
        #init = pz0
        forward_step = lambda cf, vars_t: self._forward_step(cf, vars_t)
        _, result = lax.scan(
            forward_step, init_density, (ln_pX[1:], control_z[:len(ln_pX)-1, None])
        )
        fwd_message, ln_margin_llk = result
        fwd_message = jnp.vstack((init_density, fwd_message[:,0]))
        ln_margin_llk = jnp.hstack((ln_margin_llk_0, ln_margin_llk[:,0]))
        return fwd_message, ln_margin_llk

    def _backward_step(
            self, carry: Tuple, vars_t: Tuple[int, jnp.array]
        ) -> Tuple:
        """Compute one step backward in time (smoothing).

        :param carry: Observations and control variables
        :type carry: Tuple
        :param vars_t: Data for for constructing the smoothing density of the last (future) step
        :type vars_t: Tuple
        :return: Data of new smoothing density and smoothing and two step smoothing density.
        :rtype: Tuple
        """
        t, uz_t, ln_pX_t = vars_t
        #filter_dict = {k: jnp.array(v) for k, v in filter_dict.items()}
        post_bwd_message = carry
        cur_bwd_message = self.sm.smoothing(
            post_bwd_message, u=uz_t
        )
        ln_cur_bwd_message = jnp.log(cur_bwd_message) + ln_pX_t
        ln_norm = jsc.special.logsumexp(ln_cur_bwd_message, axis=-1)
        cur_bwd_message = jnp.exp(ln_cur_bwd_message - ln_norm[..., None])
        carry = cur_bwd_message
        result = cur_bwd_message
        return carry, result

    def _backward_sweep(
        self, ln_pX: jnp.ndarray, control_z: jnp.ndarray
        ) -> jnp.ndarray:
        """Iterate backward doing smoothing step."""
        last_bwd_message = jnp.ones((self.sm.Dz,))
        cs_init = last_bwd_message
        
        backward_step = lambda cs, vars_t: self._backward_step(
            cs, vars_t
        )
        t_range = jnp.arange(0, ln_pX.shape[0])
        _, bwd_message = lax.scan(backward_step, cs_init, (t_range, control_z[:len(ln_pX), None], ln_pX[:],), reverse=True)
        bwd_messages = jnp.vstack((bwd_message, last_bwd_message[None]))
        return bwd_message
    
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
            "StationaryTransitionModel": state_model.StationaryTransitionModel,
        }
        observation_model_dict = {
            "GaussianModel": observation_model.GaussianModel,
            "ARGaussianModel": observation_model.ARGaussianModel,
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