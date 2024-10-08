__author__ = "Christian Donner"
from jax import numpy as jnp
from jax import scipy as jsc
from jax import jit, lax, vmap, random
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
        return bwd_message
    
    def get_params(self):
        """Get the parameters of the model.

        :return: Parameters of the model.
        :rtype: dict
        """
        return {"sm_params": self.sm.get_params(), "om_params": self.om.get_params()}
    
    def horizon_step(self, carry, t):
        key, pi, horizon_om = carry
        # sample the next state
        key, subkey = random.split(key)
        z = random.categorical(subkey, jnp.log(pi))
        # for AR we need to update the carried data
        key, subkey = random.split(key)
        horizon_om_new = self.om.do_one_horizon_step(horizon_om, z, subkey)
        # then we predict the next state
        pi_new = self.sm.prediction(pi)
        carry_new = (key, pi_new, horizon_om_new)
        return carry_new, None
    
    def rollout_horizon(self, key, carry_om_data: dict, pi, horizon):
        init_carry = (key, pi, carry_om_data)
        if horizon == 1:
            # if horizon is 1, no need to unroll
            last_sample = init_carry
        else:
            # otherwise we unroll the horizon (to the second last time step)
            last_sample, _ = lax.scan(self.horizon_step, init_carry, jnp.arange(horizon-1))
        pi, last_om_data = last_sample[1], last_sample[2]
        # then we get the gaussian approximation of data at the end of the horizon
        gauss_dict = self.om.get_horizon_density(pi, last_om_data)
        return {'mu': gauss_dict['mu'][0], 'Sigma': gauss_dict['Sigma'][0], 'pi_pred': pi[0]}
    
    def prediction_step(self, carry, data, observed_dims, unobserved_dims, num_samples, horizon):
        key, pi, carry_om = carry
        # get the necessary data for rolling forward the horizon (necessary for AR models, otherwise empty)
        key, subkey = random.split(key)
        init_horizon = self.om.get_initial_pred(data, carry_om, subkey, observed_dims, unobserved_dims, num_samples)
        # The horizon is rolled out and returns the last gaussian density over the data
        pred_data_dict = self.rollout_horizon(key, init_horizon, pi, horizon)
        # Observed dimensions are used for filtering, and the carried information is updated
        key, subkey = random.split(key)
        pi_filter, carry_om_new = self.om.partial_filter_step(data, pi, carry_om, observed_dims, unobserved_dims, subkey)
        # prediction for the next time step
        pi_new = self.sm.prediction(pi_filter)
        carry_new = key, pi_new, carry_om_new
        data_density = {'mu': pred_data_dict['mu'], 'Sigma': pred_data_dict['Sigma']}
        latent_density = {'pi_pred': pred_data_dict['pi_pred'], 'pi_filter': pi_filter[0]}
        return carry_new, {'x': data_density, 'z': latent_density}

    def _predict(self, X: jnp.ndarray, pi0: jnp.ndarray, control_x: jnp.ndarray, control_z: jnp.ndarray, horizon: int, 
                 observed_dims: jnp.ndarray, unobserved_dims: jnp.ndarray, 
                 first_prediction_idx: int, key: random.PRNGKey, carry_om: dict = None, num_samples: int = 1000):
        # Gets data, and lagged data in case of AR model
        data = self.om.setup_data_for_prediction(X)
        # Sets up the step func
        step_func = jit(lambda carry, data: self.prediction_step(carry, data, observed_dims, unobserved_dims, num_samples, horizon))
        # Sets up the initial carry
        if carry_om is None:
            carry_om = self.om.setup_for_partial_filtering(data, unobserved_dims)
        carry_init = (key, pi0, carry_om)
        # iterate forward and get the prediction density
        carry, res = lax.scan(step_func, carry_init, data)
        return res
    
    def predict(
        self,
        X: jnp.ndarray,
        pi0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        horizon: int = 1,
        observed_dims: jnp.ndarray = None,
        first_prediction_idx: int = 0,
        key: random.PRNGKey = random.PRNGKey(0),
    ) -> dict:
        """ Predict future states and data.

        Args:
            X: Data to predict. Dimensions should be [T, Dx] or [num_batches, T, Dx].
            pi0: Initial state density. If None, all states are similar. Defaults to None.
            control_x: TODO. Defaults to None.
            control_z: TODO. Defaults to None.
            horizon: How many steps in the future should be predicted. Defaults to 1.
            observed_dims: What dimensions are observed. Defaults to None.
            first_prediction_idx: TODO. Defaults to 0.
            key: Key for generating random numbers (only used for AR models). Defaults to random.PRNGKey(0).

        Returns:
            dict: Dictionary with the predicted data. 
            'mu' and 'Sigma' are the mean and covariance of the predicted data. 'pi' is the predicted state density.
        """
        X, pi0, control_x, control_z = self._init(
            X, pi0, control_x=control_x, control_z=control_z
        )
        if observed_dims is None:
            unobserved_dims = jnp.arange(self.om.Dx)
        elif len(observed_dims) == self.om.Dx:
            unobserved_dims = jnp.array([])
        else:    
            unobserved_dims = jnp.where(jnp.isin(jnp.arange(self.om.Dx), observed_dims, invert=True))[0]
        vmap_predict = vmap(self._predict, in_axes=(0, 0, 0, 0, None, None, None, None, 0))
        data_predict_dict = vmap_predict(X, pi0, control_x, control_z, horizon, observed_dims, unobserved_dims, first_prediction_idx, random.split(key, X.shape[0]))
        return data_predict_dict
    
    def _compute_Q_function_batch(
        self, X, pi_marg, pi_two_step, pi0, control_x, control_z
    ) -> float:
        # initial part
        initial_term = jnp.sum(pi_marg['pi'][:1] * jnp.log(jnp.maximum(pi0, 1e-10)))
        # entropy
        entropy_two_step = - jnp.sum(pi_two_step['pi'] * jnp.log(jnp.maximum(pi_two_step['pi'], 1e-10)))
        entropy_marg = - jnp.sum(pi_marg['pi'][1:] * jnp.log(jnp.maximum(pi_marg['pi'][1:], 1e-10)))
        entropy = entropy_two_step - entropy_marg
        # sm part
        sm_term = self.sm.compute_Q_function(pi_marg, pi_two_step, pi0, control_z=control_z)
        # om part
        om_term = self.om.compute_Q_function(X, pi_marg, control_x=control_x)
        # complete Q-function
        return initial_term + entropy + sm_term + om_term
    
    def compute_Q_function(self, X, pi_marg, pi_two_step, pi0, control_x=None, control_z=None) -> float:
        X, pi0, control_x, control_z = self._init(X, pi0, control_x, control_z)
        Q_batch = jit(vmap(self._compute_Q_function_batch))(
            X, pi_marg, pi_two_step, pi0, control_x, control_z
        )
        return jnp.sum(Q_batch)
        
    
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