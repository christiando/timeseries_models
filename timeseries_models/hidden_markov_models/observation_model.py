__author__ = "Christian Donner"
import scipy
from typing import Tuple
from scipy.optimize import minimize_scalar
from jax import numpy as jnp
from jax import scipy as jsc

# import numpy as np
from jax import lax
from jax import jit, grad, vmap
from gaussian_toolbox import (
    pdf,
    conditional,
    approximate_conditional,
)
from gaussian_toolbox.utils.jax_minimize_wrapper import minimize
from jax import random
from abc import abstractmethod
from functools import partial

class ObservationModel:
    def __init__(self):
        """This is the template class for observation models in state space models.
        Basically these classes should contain all functionality for the mapping between
        the latent variables z, and observations x, i.e. p(x_t|z_t). The object should
        have an attribute `observation_density`, which is be a `C, control_x, control_zonditionalDensity`.
        Furthermore, it should be possible to optimize hyperparameters, when provided
        with a density over the latent space.
        """
        self.observation_density = None
    
    def get_log_likelihoods(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError("Log likelihoods not implemented for observation model.")

    def update_hyperparameters(
        self, smoothing_density: pdf.GaussianPDF, X: jnp.ndarray, **kwargs
    ):
        """Update hyperparameters.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        raise NotImplementedError(
            "Hyperparameter updates for observation model not implemented."
        )

    def evalutate_llk(
        self, p_z: pdf.GaussianPDF, X: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Compute the log likelihood of data given distribution over latent variables.

        :param p_z: Density over latent variables.
        :type p_z: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        :return: Log likelihood
        :rtype: jnp.ndarray
        """
        raise NotImplementedError(
            "Log likelihood not implemented for observation model."
        )
        
    def get_data_density(self, pi: jnp.ndarray, **kwargs) -> pdf.GaussianPDF:
        """Get the density of the data.

        :param pi: Mass function over states.
        :type pi: jnp.ndarray [T, Dz]
        
        :return: Density of the data.
        :rtype: pdf.GaussianPDF
        """
        raise NotImplementedError("Must be implemented.")

    @staticmethod
    def _reduce_batch_dims(arrs):
        return [jnp.sum(arr, axis=0) for arr in arrs]

    @abstractmethod
    def get_params(self) -> dict:
        """Returns the parameters of the observation model."""
        raise NotImplementedError("Must be implemented.")

    @classmethod
    def from_dict(cls, params: dict):
        """Creates an observation model from a dictionary of parameters."""
        raise NotImplementedError("Must be implemented.")

    @staticmethod
    @vmap
    def mat_to_cholvec(mat: jnp.ndarray) -> jnp.ndarray:
        """Converts a lower triangular matrix to a vector.

        :param mat: Lower triangular matrix.
        :type mat: jnp.ndarray
        :return: Vectorized lower triangular matrix.
        :rtype: jnp.ndarray
        """
        L = jnp.linalg.cholesky(mat)
        vec = L[jnp.tril_indices_from(L)]
        return vec

    @staticmethod
    @vmap
    def cholvec_to_mat(vec: jnp.ndarray, n_dim: int) -> jnp.ndarray:
        """Converts a vectorized lower triangular matrix to a matrix.

        :param vec: Vectorized lower triangular matrix.
        :type vec: jnp.ndarray
        :param n_dim: Dimensionality of matrix.
        :type n_dim: int
        :return: Matrix.
        :rtype: jnp.ndarray
        """
        L = jnp.zeros((n_dim, n_dim))
        M = L.at[jnp.tril_indices_from(L)].set(vec)
        return M @ M.T

class GaussianModel(ObservationModel):
    def __init__(self, Dx: int, Dz: int, noise_x: float = 1., key: random.PRNGKey = None):
        self.Dx, self.Dz = Dx, Dz
        self.noise_x = noise_x
        if key is None:
            key = random.PRNGKey(0)
        self.mu = jnp.zeros((Dz, Dx)) + 1e-2 * random.normal(key, (Dz, Dx))
        self.Sigma = jnp.tile(jnp.eye(Dx)[None] * noise_x, (Dz, 1, 1))
        self.L = self.mat_to_cholvec(self.Sigma)
        self.observation_density = pdf.GaussianPDF(mu=self.mu, Sigma=self.Sigma)
        
        
    def get_log_likelihoods(self, X: pdf.Array, **kwargs) -> jnp.ndarray:
        return self.observation_density.evaluate_ln(X).T
    
    def update_hyperparameters(self, X: jnp.ndarray, pi_marg: dict, **kwargs):
        pi = pi_marg['pi']
        sum_pi = jnp.sum(jnp.sum(pi, axis=0), axis=0)
        dX_mu = X[:,:,None] - self.mu[None, None]
        self.Sigma = jnp.einsum('abcd, abce -> cde', dX_mu, dX_mu * pi[:,:,:,None]) / sum_pi[:,None, None]
        self.mu = jnp.einsum('abc, abd -> cd', pi, X) / sum_pi[:,None]
        self.L = self.mat_to_cholvec(self.Sigma)
        self.observation_density = pdf.GaussianPDF(mu=self.mu, Sigma=self.Sigma)
    
    def partial_filter_step(self, densities: dict, data: dict, observed_dims: jnp.ndarray=None, unobserved_dims: jnp.ndarray=None, **kwargs) -> jnp.ndarray:
        pi = densities['pi']
        if observed_dims is not None:
            X = data['X']
            pX = self.observation_density.get_marginal(observed_dims)
            ln_pX = pX.evaluate_ln(X[..., observed_dims])
            ln_pi_filtered = jnp.log(pi) + ln_pX.T
            pi_filtered = jnp.exp(ln_pi_filtered - jsc.special.logsumexp(ln_pi_filtered, axis=-1, keepdims=True))
        else:
            pi_filtered = pi
        densities = {'pi': pi_filtered}
        return densities
    
    def setup_data_for_prediction(self, X: jnp.ndarray, control_x: jnp.ndarray=None, control_z: jnp.ndarray=None, **kwargs) -> dict:
        return {'X': X[:,None]}
    
    def get_data_density(self, densities: dict, data: dict, observed_dims: jnp.ndarray=None, unobserved_dims: jnp.ndarray=None, **kwargs) -> pdf.GaussianPDF:
        X = data['X']
        pi = densities['pi']
        if observed_dims is not None:
            pX_u_given_o = self.observation_density.condition_on_explicit(observed_dims, unobserved_dims)(X[..., observed_dims])
        else:
            pX_u_given_o = self.observation_density
        #print(pi.shape, pX_u_given_o.mu.shape)
        mu = jnp.einsum('ab, bc -> ac', pi, pX_u_given_o.mu)
        Sigma = jnp.einsum('ab, bcd -> acd' , pi, pX_u_given_o.integrate("xx'")) - jnp.einsum('ab, ac -> abc', mu, mu)
        return pdf.GaussianPDF(mu=mu, Sigma=Sigma)
        
        
    def get_params(self) -> dict:
        params = {'mu': self.mu, 'L': self.L}
        return params
    
    @classmethod
    def from_dict(cls, params: dict):
        Dx, Dz = params['mu'].shape[1], params['mu'].shape[0]
        model = cls(Dx, Dz)
        model.mu = params['mu']
        model.L = params['L']
        model.Sigma = cls.cholvec_to_mat(model.L, Dx)
        model.observation_density = pdf.GaussianPDF(mu=model.mu, Sigma=model.Sigma)
        return model
    
class ARGaussianModel(ObservationModel):
    def __init__(self, Dx: int, Dz: int, lags:int=1, noise_x: float = 1, key: random.PRNGKey = None):
        self.Dx, self.Dz = Dx, Dz
        self.lags = lags
        self.noise_x = noise_x
        M = jnp.concatenate([jnp.eye(Dx), jnp.zeros((Dx, Dx * (self.lags - 1)))], axis=1)
        self.M = jnp.tile(M[None], (Dz, 1, 1))
        if key is None:
            key = random.PRNGKey(0)
        self.b = random.normal(key, (Dz, Dx))
        self.Sigma = jnp.tile(jnp.eye(Dx)[None] * noise_x, (Dz, 1, 1))
        self.observation_density = conditional.ConditionalGaussianPDF(M=self.M, b=self.b, Sigma=self.Sigma)
        
    def _create_lagged_data(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        T = X.shape[0]
        X_lagged = []
        for i in range(self.lags):
            X_lagged.append(X[i:T - self.lags + i][:, None])
        X_lagged = jnp.concatenate(X_lagged, axis=1)[:,::-1]
        X_lagged = X_lagged.reshape((T - self.lags, self.lags * self.Dx))
        return X_lagged

    def get_lagged_dims(self, dims: jnp.ndarray):
        lagged_dims = []
        for i in range(self.lags):
            lagged_dims.append(dims + i * self.Dx)
        return jnp.concatenate(lagged_dims)
    
    @partial(vmap, in_axes=(None, 0, 0))
    def _condition_and_eval(self, X_lagged: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        return self.observation_density(X_lagged[None]).evaluate_ln(X[None])[:,0]
    
    def partial_filter_step(self, densities: dict, data: dict,
                            observed_dims: jnp.ndarray=None, unobserved_dims: jnp.ndarray=None, **kwargs) -> jnp.ndarray:
        pi, p_X_lagged = densities['pi'], densities['p_X_lagged']
        if observed_dims is not None:
            X, X_lagged = data['X'], data['X_lagged']
            lagged_observed_dims = self.get_lagged_dims(observed_dims)
            lagged_unobserved_dims = self.get_lagged_dims(unobserved_dims)
            M_new = self.observation_density.M[..., lagged_unobserved_dims]
            b_new = self.observation_density.b + jnp.einsum('abc, c -> ab', self.observation_density.M[..., lagged_observed_dims], X_lagged[..., lagged_observed_dims])
            new_observation_density = conditional.ConditionalGaussianPDF(M=M_new, b=b_new, 
                                                                         Sigma=self.observation_density.Sigma, 
                                                                         L=self.observation_density.L, 
                                                                         ln_det_Sigma=self.observation_density.ln_det_Sigma)
            pX = new_observation_density.affine_marginal_transformation(p_X_lagged)
            pX_observed = pX.get_marginal(observed_dims)
            ln_pX = pX_observed.evaluate_ln(X[..., observed_dims]) 
            # pX = self.observation_density.get_marginal(observed_dims)
            ln_pX = pX.evaluate_ln(X[..., observed_dims])
            ln_pi_filtered = jnp.log(pi) + ln_pX
            pi_filtered = jnp.exp(ln_pi_filtered - jsc.special.logsumexp(ln_pi_filtered, axis=-1, keepdims=True))
            pX_X_lagged = new_observation_density.affine_joint_transformation(p_X_lagged)
            pX_X_lagged_observed = pX_X_lagged.condition_on(observed_dims)(X[..., observed_dims])
            pX_X_lagged_new = pX_X_lagged_observed.get_marginal(jnp.arange(p_X_lagged.Dx))
        else:
            pi_filtered = pi
            pX_X_lagged = new_observation_density.affine_joint_transformation(p_X_lagged)
            pX_X_lagged_new = pX_X_lagged.get_marginal(jnp.arange(p_X_lagged.Dx))
            
        mu = jnp.einsum('ab, bc -> ac', pi_filtered, pX_X_lagged_new.mu)
        Sigma = jnp.einsum('ab, bcd -> acd' , pi_filtered, pX_X_lagged_new.integrate("xx'")) - jnp.einsum('ab, ac -> abc', mu, mu)
        px_lagged_gauss = pdf.GaussianPDF(mu=mu, Sigma=Sigma)
        densities = {'pi': pi_filtered, 'p_X_lagged': px_lagged_gauss}
            
        return pi_filtered
    
    def get_data_density(self, pi: jnp.ndarray, X: jnp.ndarray, observed_dims: jnp.ndarray=None, **kwargs) -> pdf.GaussianPDF:
        if observed_dims is not None:
            pX_u_given_o = self.observation_density.condition_on(observed_dims)(X[..., observed_dims])
        else:
            pX_u_given_o = self.observation_density
        mu = jnp.einsum('ab, bc -> ac', pi, pX_u_given_o.mu)
        Sigma = jnp.einsum('ab, bcd -> acd' , pi, pX_u_given_o.integrate("xx'")) - jnp.einsum('ab, ac -> abc', mu, mu)
        return pdf.GaussianPDF(mu=mu, Sigma=Sigma)
        
    
    def partial_filter_step(self, X: jnp.ndarray, observed_dims: jnp.ndarray, past_info: dict, **kwargs) -> pdf.GaussianPDF:
        gaussian_observation_density = self._get_gaussian_observation_density(past_info)
        return gaussian_observation_density.condition_on(observed_dims)(X[..., observed_dims])
        
    
    def get_log_likelihoods(self, X: pdf.Array, **kwargs) -> jnp.ndarray:
        X_lagged = self._create_lagged_data(X)
        return self._condition_and_eval(X_lagged, X[self.lags:])
    
    def update_hyperparameters(self, X: jnp.ndarray, pi_marg: dict, **kwargs):
        pi = pi_marg['pi']
        sum_pi = jnp.sum(jnp.sum(pi, axis=0), axis=0)
        X_lagged = vmap(self._create_lagged_data)(X)
        mu_t = jnp.swapaxes(vmap(self.observation_density.get_conditional_mu)(X_lagged), 1, 2)
        dX_mu = X[:, self.lags:, None,:] - mu_t
        self.Sigma = jnp.einsum('abcd, abce -> cde', dX_mu, dX_mu * pi[:,:,:,None]) / sum_pi[:,None, None]
        self.L = self.mat_to_cholvec(self.Sigma)
        dX_FX = jnp.einsum('abcd, abc -> cd', X[:, self.lags:,None] - jnp.einsum('abc, dec -> abde', X_lagged, self.M), pi)
        self.b = dX_FX / sum_pi[:,None]
        dX_b_vec = jnp.einsum('abcd, abce -> ced', X[:, self.lags:, None] - self.b[None, None], X_lagged[:,:,None] * pi[...,None] )
        xx_lagged = jnp.einsum('abc, abde -> dce', X_lagged, X_lagged[:,:,None] * pi[...,None])
        self.M = jnp.swapaxes(vmap(jnp.linalg.solve)(xx_lagged, dX_b_vec), -1, -2)
        self.observation_density = conditional.ConditionalGaussianPDF(M=self.M, b=self.b, Sigma=self.Sigma)
        
    def get_data_density(self, pi: jnp.ndarray, p_past: pdf.GaussianPDF, **kwargs) -> pdf.GaussianPDF:
        observation_density = self.observation_density.affine_marginal_transformation(p_past)
        mu = jnp.einsum('ab, bc -> ac', pi, observation_density.mu)
        Sigma = jnp.einsum('ab, bcd -> acd' , pi, observation_density.integrate("xx'")) - jnp.einsum('ab, ac -> abc', mu, mu)
        return pdf.GaussianPDF(mu=mu, Sigma=Sigma)
    
    def get_params(self) -> dict:
        return {'M': self.M, 'b': self.b, 'L': self.L}
    
    @classmethod
    def from_dict(cls, params: dict):
        Dz, Dx = params['M'].shape[0], params['M'].shape[1]
        lags = params['M'].shape[2] // Dx
        model = cls(Dx, Dz, lags)
        model.M = params['M']
        model.b = params['b']
        model.Sigma = cls.cholvec_to_mat(params['L'], Dx)
        model.L = params['L']
        model.observation_density = conditional.ConditionalGaussianPDF(M=model.M, b=model.b, Sigma=model.Sigma)
        return model
        
            
    