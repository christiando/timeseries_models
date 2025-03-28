__author__ = "Christian Donner"
from jax import numpy as jnp
from jax import jit, vmap
from timeseries_models import observation_model, state_model
from timeseries_models.utils.em_util_funcs import _estep, _mstep, _predict
from gaussian_toolbox import pdf
import pickle
import os
import time
##################################################################################################
# This file is part of the Gaussian Toolbox,                                                     #
#                                                                                                #
# It contains the class to fit state space models (SSMs) with the expectation-maximization       #
# algorithm.                                                                                     #
#                                                                                                #
# Author: Christian Donner                                                                       #
##################################################################################################

__author__ = "Christian Donner"
from typing import Tuple
from tqdm import tqdm


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
        
        self.params = self.get_params()
        
    def _init_models(self, params):
        sm = self.sm.from_dict(params['sm_params'])
        om = self.om.from_dict(params['om_params'])  
        return sm, om 
    
    def _init(
        self,
        X: jnp.ndarray,
        mu0: jnp.ndarray = None,
        Sigma0: jnp.ndarray = None,
        control_x: jnp.ndarray = None,
        control_z: jnp.ndarray = None,
        horizon: int = 1,
    ) -> Tuple[jnp.ndarray]:
        assert X.shape[-1] == self.om.Dx
        assert control_x is None or control_x.ndim == X.ndim
        assert control_z is None or control_z.ndim == X.ndim
        if X.ndim == 2:
            X = X[None]
        if control_x is None:
            control_x = jnp.empty((X.shape[0], X.shape[1] + horizon - 1, 0))
        elif control_x.ndim == 2:
            control_x = control_x[None]
        if control_z is None:
            control_z = jnp.empty((X.shape[0], X.shape[1] + horizon - 1, 0))
        elif control_z.ndim == 2:
            control_z = control_x[None]
        assert control_x.shape[1] == (X.shape[1] + horizon - 1)
        assert control_z.shape[1] == (X.shape[1] + horizon - 1)
        if mu0 is None:
            mu0 = jnp.zeros((X.shape[0], 1, self.sm.Dz))
        if Sigma0 is None:
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
        llk_list = []
        llk_old = -jnp.inf
        # instead of print use tqdm
        
        pbar = tqdm(total=max_iter, desc="EM Algorithm", dynamic_ncols=True)
        
        while iteration < max_iter and not converged:
            time_start_total = time.perf_counter()
            smooth_dict, two_step_smooth_dict = self.estep(
                X, mu0, Sigma0, control_x, control_z
            )
            mu0, Sigma0 = self.mstep(
                X, smooth_dict, two_step_smooth_dict, control_x, control_z
            )
            self.sm, self.om = self._init_models(self.params)
            llk = self.compute_predictive_log_likelihood(X, mu0, Sigma0, control_x, control_z)
            llk_list.append(llk)
            if iteration > 2:
                converged = self._check_convergence(llk_list[-2], llk, conv_crit)
            # if iteration % 1 == 0:
            #     print("Iteration %d - Log likelihood=%.1f" % (iteration, llk_old))
            tot_time = time.perf_counter() - time_start_total
            # if timeit:
            #     print(
            #         "###################### \n"
            #         + "E-step: Run Time %.1f \n" % etime
            #         + "LLK-func: Run Time %.1f \n" % llk_time
            #         + "M-step: Run Time %.1f \n" % mtime
            #         + "Total: Run Time %.1f \n" % tot_time
            #         + "###################### \n"
            #     )
            pbar.set_postfix(iteration=iteration, log_likelihood=llk, total_time=f"{tot_time:.1f}s")
            pbar.update(1)
            iteration += 1
        if not converged:
            print("EM reached the maximal number of iterations.")
        else:
            print("EM did converge.")
        p0_dict = {"Sigma": Sigma0, "mu": mu0}
        return llk_list, p0_dict, smooth_dict, two_step_smooth_dict

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
        return_as_dict: bool = False,
    ):
        X, mu0, Sigma0, control_x, control_z = self._init(
            X,
            mu0=mu0,
            Sigma0=Sigma0,
            control_x=control_x,
            control_z=control_z,
            horizon=horizon,
        )
        predict_func = lambda X, mu0, Sigma0, control_x, control_z: _predict(
                    self.params,
                    self._init_models,
                    X,
                    mu0,
                    Sigma0,
                    control_x,
                    control_z,
                    horizon,
                    observed_dims,
                    first_prediction_idx,
                )
        
        data_predict_dict, latent_predict_dict = predict_func(X, mu0, Sigma0, control_x, control_z)
        if return_as_dict:
            return data_predict_dict, latent_predict_dict
        else:
            data_prediction_densities = []
            latent_prediction_densities = []
            num_batches = X.shape[0]
            for ibatch in range(num_batches):
                batch_density = pdf.GaussianPDF(
                    Sigma=data_predict_dict["Sigma"][ibatch],
                    mu=data_predict_dict["mu"][ibatch],
                    Lambda=data_predict_dict["Lambda"][ibatch],
                    ln_det_Sigma=data_predict_dict["ln_det_Sigma"][ibatch],
                )
                data_prediction_densities.append(batch_density)
                latent_density = pdf.GaussianPDF(
                    Sigma=latent_predict_dict["Sigma"][ibatch],
                    mu=latent_predict_dict["mu"][ibatch],
                    Lambda=latent_predict_dict["Lambda"][ibatch],
                    ln_det_Sigma=latent_predict_dict["ln_det_Sigma"][ibatch],
                )
                latent_prediction_densities.append(latent_density)
            if num_batches == 1:
                return {'x': data_prediction_densities[0], 'z': latent_prediction_densities[0]}
            else:
                return {'x': data_prediction_densities, 'z': latent_prediction_densities}

    def mstep(
        self, X, smooth_dict, two_step_smooth_dict, control_x, control_z
    ) -> Tuple[jnp.ndarray]:
        """Perform the maximization step, i.e. the updates of model parameters."""
        # Update parameters of state model
        self.params = _mstep(self.params, self._init_models, X, smooth_dict, two_step_smooth_dict, control_x, control_z)
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
        return _estep(self.params, self._init_models, X, mu0, Sigma0, control_x, control_z)

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
        #phi = smoothing_density.slice(jnp.arange(0, T))
        om_Q = self.om.compute_Q_function(smoothing_density, X, control_x=control_x)
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
        )['x']
        llk = 0
        num_batches = X.shape[0]
        if num_batches == 1:
            llk += jnp.sum(
                predictive_densities.evaluate_ln(
                    X[0, first_prediction_idx:], element_wise=True
                )[ignore_init_samples:]
            )
        else:
            for ibatch, density in enumerate(predictive_densities):
                llk += jnp.sum(
                    density.evaluate_ln(X[ibatch, first_prediction_idx:], element_wise=True)[
                        ignore_init_samples:
                    ]
                )
        return llk

    # TODO: fix this
    # def _sample_step(
    #     self,
    #     z_old: jnp.array,
    #     vars_t: Tuple,
    #     observed_dims: jnp.ndarray,
    #     unobserved_dims: jnp.ndarray,
    # ) -> Union[jnp.ndarray, jnp.ndarray]:
    #     """One time step sample for fixed observed data dimensions.

    #     :param z_old: Sample of latent variable in previous time step.
    #     :type z_old: jnp.ndarray [num_samples, Dz]
    #     :param rand_nums_z: Random numbers for sampling latent dimensions.
    #     :type rand_nums_z: jnp.ndarray [num_samples, Dz]
    #     :param x: Data vector for current time step.
    #     :type x: jnp.ndarray [1, Dx]
    #     :param rand_nums_x: Random numbers for sampling x.
    #     :type rand_nums_x: jnp.ndarray [num_samples, num_unobserved_dims]
    #     :param observed_dims: Observed dimensions.
    #     :type observed_dims: jnp.ndarray [num_observed_dims]
    #     :param unobserved_dims: Unobserved dimensions.
    #     :type unobserved_dims: jnp.ndarray [num_unobserved_dims]
    #     :return: Latent variable and data sample (only unobserved) for current time step.
    #     :rtype: Union[jnp.ndarray, jnp.ndarray] [num_samples, Dz] [num_samples, num_unobserved]
    #     """

    #     rand_nums_z_t, x_t, rand_nums_x_t, uz_t, ux_t = vars_t
    #     p_z = self.sm.condition_on_past(z_old, u=uz_t)
    #     L = jnp.linalg.cholesky(p_z.Sigma)
    #     z_sample = p_z.mu + jnp.einsum("abc,ac->ab", L, rand_nums_z_t)
    #     p_x = self.om.condition_on_z_and_observations(
    #         z_sample, x_t, observed_dims, unobserved_dims, ux_t=ux_t
    #     )
    #     L = jnp.linalg.cholesky(p_x.Sigma)
    #     x_sample = p_x.mu + jnp.einsum("abc,ac->ab", L, rand_nums_x_t)
    #     result = z_sample, x_sample
    #     return z_sample, result

    # def sample_trajectory(
    #     self,
    #     X: jnp.ndarray,
    #     observed_dims: jnp.ndarray = None,
    #     p0: pdf.GaussianPDF = None,
    #     num_samples: int = 1,
    #     u_z: jnp.ndarray = None,
    #     u_x: jnp.ndarray = None,
    # ) -> Union[jnp.ndarray, jnp.ndarray]:
    #     """Samples a trajectories, with fixed observed data dimensions.

    #     :param X: Data array containing the variabels to condition on, and indicating how long we wish to sample.
    #     :type X: jnp.ndarray [T, Dx]
    #     :param observed_dims: Dimension that are observed. If none no dimension is observed, defaults to None
    #     :type observed_dims: jnp.ndarray, optional [num_observed_dimensions]
    #     :param p0: Initial state density. If none, standard normal., defaults to None
    #     :type p0: pdf.GaussianPDF, optional
    #     :param num_samples: How many trajectories should be sampled, defaults to 1
    #     :type num_samples: int, optional
    #     :return: Samples of the latent variables, and the unobserved data dimensions.
    #     :rtype: Union[jnp.ndarray, jnp.ndarray] [T+1, nums_samples, Dz] [T, nums_samples, num_unobserved_dims]
    #     """
    #     T = X.shape[0]
    #     if u_z is None:
    #         u_z = jnp.empty((T, 0))
    #     if u_x is None:
    #         u_x = jnp.empty((T, 0))
    #     if p0 is None:
    #         p0 = pdf.GaussianPDF(
    #             Sigma=jnp.array([jnp.eye(self.sm.Dz)]), mu=jnp.zeros((1, self.sm.Dz))
    #         )
    #     if observed_dims is None:
    #         unobserved_dims = jnp.arange(self.om.Dx)
    #         num_unobserved_dims = X.shape[1]
    #     else:
    #         unobserved_dims = jnp.setxor1d(jnp.arange(self.om.Dx), observed_dims)
    #         num_unobserved_dims = len(unobserved_dims)

    #     init = jnp.asarray(p0.sample(num_samples)[:, 0])
    #     sample_step = jit(
    #         lambda z_old, vars_t: self._sample_step(
    #             z_old, vars_t, observed_dims, unobserved_dims
    #         )
    #     )
    #     # TODO: fix this
    #     rand_nums_z = objax.random.normal((T, num_samples, self.Dz))
    #     rand_nums_x = objax.random.normal((T, num_samples, num_unobserved_dims))

    #     _, result = lax.scan(
    #         sample_step, init, (rand_nums_z, X, rand_nums_x, u_z[:, None], u_x[:, None])
    #     )
    #     z_sample, X_sample = result

    #     return z_sample, X_sample

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
            #params = self.get_params()
            #model_dict = params | {
            #    "sm_class": self.sm.__class__.__name__,
            #    "om_class": self.om.__class__.__name__,
            #}
            pickle.dump(self, open(f"{path}/{model_name}.p", "wb"))

    @classmethod
    def load(cls, model_name: str, path: str = ""):
        """Load the model.

        :param model_name: Name of the model, which is used as file name.
        :type model_name: str
        :param path:  Path to which model is saved to, defaults to ""
        :type path: str, optional
        """
        model = pickle.load(open("%s/%s.p" % (path, model_name), "rb"))
        return model
