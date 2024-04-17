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

class GaussianObservationModel(ObservationModel):
    def __init__(self, Dx: int, Dz: int, noise_x: float = 1., key: random.PRNGKey = None):
        self.Dx, self.Dz = Dx, Dz
        self.noise_x = noise_x
        if key is None:
            key = random.PRNGKey(0)
        self.mu = jnp.zeros((Dz, Dx)) + 1e-4 * random.normal(key, (Dz, Dx))
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
        self.b = random.normal(key, (Dz, Dx * self.lags))
        self.Sigma = jnp.tile(jnp.eye(Dx)[None] * noise_x, (Dz, 1, 1))
        self.observation_density = conditional.ConditionalGaussianPDF(M=self.M, b=self.b, Sigma=self.Sigma)
        
    def _create_lagged_data(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        T = X.shape[0]
        X_lagged = []
        for i in range(self.lags):
            X_lagged.append(X[i:T - self.lags + i][:, None])
        X_lagged = jnp.concatenate(X_lagged, axis=1)[:,::-1]
        X_lagged = X_lagged.reshape((T - self.lags + 1, self.lags * self.Dx))
        return X_lagged
    
    @vmap
    def _condition_and_eval(self, X_lagged: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        return self.observation_density(X_lagged[:,None]).evaluate_ln(X[:,None])[:,0]
    
    def get_log_likelihoods(self, X: pdf.Array, **kwargs) -> jnp.ndarray:
        X_lagged = self._create_lagged_data(X)
        return self._condition_and_eval(X_lagged, X)
    
    def update_hyperparameters(self, X: jnp.ndarray, pi_marg: dict, **kwargs):
        X_lagged = self._create_lagged_data(X)
        
    
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
        
            
    