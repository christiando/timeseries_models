__author__ = "Christian Donner"
from jax import numpy as jnp
import jax
from jax import vmap, jit, random, lax
from typing import Tuple
from gaussian_toolbox.utils.jax_minimize_wrapper import minimize

from gaussian_toolbox import (
    pdf,
    conditional,
    approximate_conditional,
    factor,
)
from abc import abstractmethod


class StateModel:
    """This is the template class for state transition models in state space models.
    Basically these classes should contain all functionality for transition between time steps
    the latent variables :math:`Z`, i.e. :math:`p(Z_{t+1}|Z_t)`. The object should
    have an attribute `state_density`, which is be a `ConditionalDensity`.
    Furthermore, it should be possible to optimize hyperparameters, when provided
    with a density over the latent space.
    """

    def __init__(self, Dz: int):
        self.Dz = Dz
        
    def get_transition_matrix(self, **kwargs) -> jnp.ndarray:
        """Get the transition matrix of the state model.
        
        :raises NotImplementedError: Must be implemented.
        """
        raise NotImplementedError("Must be implemented.")

    def prediction(
        self, pre_pi: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        r"""Calculate prediction density.

        .. math::

            p(Z_t|X_{1:t-1}) = \sum_{Z_{t-1}} p(Z_t|Z_{t-1})p(Z_{t-1}|X_{1:t-1}).

        :param pre_pi: Mass function :math:`p(Z_{t-1}|X_{1:t-1})`.
        :type pre_pi: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        :return: Prediction mass function :math:`p(Z_t|X_{1:t-1}).
        :rtype: jnp.ndarray
        """
        raise NotImplementedError("Prediction for state model not implemented.")

    def smoothing(
        self,
        cur_filter_density: pdf.GaussianPDF,
        post_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ) -> Tuple[pdf.GaussianPDF, pdf.GaussianPDF]:
        """Calculate smoothing density :math:`p(Z_{t} | X_{1:T})`,
        given :math:`p(Z_{t+1} | X_{1:T})` and :math:`p(Z_{t} | X_{1:t})`.

        :param cur_filter_density: Density :math:`p(Z_t|X_{1:t})`
        :type cur_filter_density: pdf.GaussianPDF
        :param post_smoothing_density: Density :math:`p(Z_{t+1}|X_{1:T})`
        :type post_smoothing_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        :return: Smoothing density :math:`p(Z_t|X_{1:T})` and :math:`p(Z_{t+1}, Z_t|X_{1:T})`
        :rtype: Tuple[pdf.GaussianPDF, pdf.GaussianPDF]
        """
        raise NotImplementedError("Smoothing for state model not implemented.")

    def update_hyperparameters(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ):
        """The hyperparameters are updated here, where the the densities :math:`p(Z_t|X_{1:T})` and
        :math:`p(Z_{t+1}, Z_t|x_{1:T})` are provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        """
        raise NotImplementedError("Hyperparamers for state model not implemented.")

    @staticmethod
    def _reduce_batch_dims(arrs):
        return [jnp.sum(arr, axis=0) for arr in arrs]

    @abstractmethod
    def get_params(self) -> dict:
        """Get the parameters of the state model.

        :raises NotImplementedError: Must be implemented.
        """
        raise NotImplementedError("Must be implemented.")

    @abstractmethod
    def from_dict(cls, params: dict):
        raise NotImplementedError("Must be implemented.")
    

class StationaryTransitionModel(StateModel):
    
    def __init__(self, Dz: int):
        super().__init__(Dz)
        self.transition_matrix = 0.9 * jnp.eye(self.Dz) + 0.1
        self.transition_matrix /= jnp.sum(self.transition_matrix, axis=0)[None]
        
    def get_transition_matrices(self, **kwargs) -> jnp.ndarray:
        return self.transition_matrix[None]
    
    def prediction(
        self, pre_pi: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        return jnp.dot(pre_pi, self.transition_matrix.T)
    
    def smoothing(self, post_pi: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.dot(post_pi, self.transition_matrix)
    
    def update_hyperparameters(self, pi_marg: dict, pi_two_step: dict, **kwargs):
        transmat = jnp.sum(jnp.sum(pi_two_step['pi'], axis=0), axis=0).T + 1e-8
        transmat /= jnp.sum(transmat, axis=0)[None]
        self.transition_matrix = transmat
        
    def get_params(self) -> dict:
        return {'transition_matrix': self.transition_matrix}
    
    @classmethod
    def from_dict(cls, params: dict):
        transition_matrix = params['transition_matrix']
        Dz = transition_matrix.shape[0]
        model = cls(Dz)
        model.transition_matrix = params['transition_matrix']
        return model
        

