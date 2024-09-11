__author__ = "Christian Donner"
from jax import numpy as jnp
import jax
from jax import vmap, jit, random, lax
import haiku as hk
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

    def __init__(self):
        self.state_density = None

    def prediction(
        self, pre_filter_density: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Calculate prediction density.

        .. math::

            p(Z_t|X_{1:t-1}) = \int p(Z_t|Z_{t-1})p(Z_{t-1}|X_{1:t-1}) {\rm d}Z_{t-1}.

        :param pre_filter_density: Density :math:`p(Z_{t-1}|X_{1:t-1})`.
        :type pre_filter_density: pdf.GaussianPDF
        :raises NotImplementedError: Must be implemented.
        :return: Prediction density :math:`p(Z_t|X_{1:t-1}).
        :rtype: pdf.GaussianPDF
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

    @staticmethod
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


class LinearStateModel(StateModel):
    def __init__(self, Dz: int, noise_z: float = 1.0, delta: float = 0.0):
        r"""This implements a linear state transition model

        .. math::

            Z_t = A Z_{t-1} + b + \zeta_t \text{ with } \zeta_t \sim {\cal N}(0, \Sigma_z).

        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param noise_z: Intial isoptropic std. on the state transition, defaults to 1.0
        :type noise_z: float, optional
        """
        self.Dz = Dz
        self.Qz = noise_z**2 * jnp.eye(self.Dz)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        self.A, self.b = jnp.eye(self.Dz), jnp.zeros((self.Dz,))
        self.delta = delta
        self.update_state_density()

    def prediction(
        self, pre_filter_density: pdf.GaussianPDF, **kwargs
    ) -> pdf.GaussianPDF:
        r"""Calculate the prediction density.

        .. math::

            p(Z_t|X_{1:t-1}) = \int p(Z_t|Z_{t-1})p(Z_{t-1}|X_{1:t-1}) {\rm d}Z_{t-1}

        :param pre_filter_density: Density :math:`p(z_t-1|x_{1:t-1})`
        :type pre_filter_density: pdf.GaussianPDF
        :return: Prediction density :math:`p(z_t|x_{1:t-1})`.
        :rtype: pdf.GaussianPDF
        """
        # p(z_t|x_{1:t-1})
        return self.state_density.affine_marginal_transformation(
            pre_filter_density, **kwargs
        )

    def smoothing(
        self,
        cur_filter_density: pdf.GaussianPDF,
        post_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ) -> Tuple[pdf.GaussianPDF, pdf.GaussianPDF]:
        r"""Perform smoothing step.

        First we calculate the backward density

        .. math::

            p(Z_{t} | Z_{t+1}, X_{1:t}) = \frac{p(Z_{t+1}|Z_t)p(Z_t | X_{1:t})}{p(Z_{t+1}| X_{1:t})}


        and finally we get the smoothing density

        .. math::

            p(Z_{t} | X_{1:T}) = \int p(Z_{t} | Z_{t+1}, X_{1:t}) p(Z_{t+1}|X_{1:T}) {\rm d}Z_{t+1}


        :param cur_filter_density: Density :math:`p(Z_t|X_{1:t})`
        :type cur_filter_density: pdf.GaussianPDF
        :param post_smoothing_density: Density :math:`p(Z_{t+1}|X_{1:T})`
        :type post_smoothing_density: pdf.GaussianPDF
        :return: Smoothing density :math:`p(Z_t|X_{1:T})` and :math:`p(Z_{t+1}, Z_t|X_{1:T})`
        :rtype: Tuple[pdf.GaussianPDF, pdf.GaussianPDF]
        """
        # p(z_{t} | z_{t+1}, x_{1:t})
        backward_density = self.state_density.affine_conditional_transformation(
            cur_filter_density, **kwargs
        )
        # p(z_{t}, z_{t+1} | x_{1:T})
        cur_two_step_smoothing_density = backward_density.affine_joint_transformation(
            post_smoothing_density
        )
        # p(z_{t} | x_{1:T})
        cur_smoothing_density = cur_two_step_smoothing_density.get_marginal(
            jnp.arange(self.Dz, 2 * self.Dz)
        )

        return cur_smoothing_density, cur_two_step_smoothing_density

    def compute_Q_function(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        **kwargs
    ) -> float:
        r"""Compute state part of the Q-function.

        .. math::

            \sum \mathbb{E}[\ln p(Z_t|Z_{t-1})],

        where the expection is over the smoothing density.

        :param smoothing_density: Smoothing density :math:`p(Z_t|X_{1:T})`
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        :return: Evaluated state part of the Q-function.
        :rtype: float
        """
        return jnp.sum(
            self.state_density.integrate_log_conditional(
                two_step_smoothing_density,
                p_x=smoothing_density.slice(jnp.arange(0, smoothing_density.R-1)),
            )
        )

    def update_hyperparameters(
        self, smooth_dict: dict, two_step_smooth_dict: dict, **kwargs
    ):
        """Update hyperparameters.

        The densities :math:`p(Z_t|X_{1:T})` and :math:`p(Z_{t+1}, Z_t|X_{1:T})` need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density  :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        self.A = jit(self._update_A)(smooth_dict, two_step_smooth_dict, **kwargs)
        self.Qz = jit(self._update_Qz)(two_step_smooth_dict, **kwargs)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        self.b = jit(self._update_b)(smooth_dict, **kwargs)
        self.update_state_density()

    def _update_A(self, smooth_dict: dict, two_step_smooth_dict: dict, **kwargs):
        """Update transition matrix.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        # Ezz = smoothing_density.integrate("xx'")
        stats = vmap(self._get_A_stats)(smooth_dict, two_step_smooth_dict)
        A, b = self._reduce_batch_dims(stats)
        A_inv = jnp.linalg.pinv(A)
        #print(b.shape, A_inv.shape)
        A_new = jnp.dot(b.T, A_inv)
        return A_new

    def _get_A_stats(self, smooth_dict: dict, two_step_smooth_dict: dict):
        # T = smooth_dict["mu"].shape[0]
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        mu_b = smoothing_density.mu[:-1, :, None] * self.b[None, None]
        Ezz_two_step = two_step_smoothing_density.integrate("xx'")
        #Ezz = Ezz_two_step[:, self.Dz :, self.Dz :]
        Ezz = smoothing_density.integrate("xx'")[:-1]
        Ezz_cross = Ezz_two_step[:, self.Dz :, : self.Dz]
        A = jnp.mean(Ezz, axis=0)  # + 1e-2 * jnp.eye(self.Dz)
        b = jnp.mean(Ezz_cross - mu_b, axis=0)
        return A, b

    def _update_b(self, smooth_dict: dict, **kwargs):
        """Update transition offset.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        """
        stats = vmap(self._get_b_stats)(smooth_dict)
        T, b = self._reduce_batch_dims(stats)
        return b / T

    def _get_b_stats(self, smooth_dict: dict):
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        T = smoothing_density.R - 1
        b_vec = jnp.sum(
            smoothing_density.mu[1:] - jnp.dot(self.A, smoothing_density.mu[:-1].T).T,
            axis=0,
        )
        return jnp.array([T]), b_vec

    def _update_Qz(self, two_step_smooth_dict: pdf.GaussianPDF, **kwargs):
        """Update transition covariance.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        stats = vmap(self._get_Qz_stats)(two_step_smooth_dict)
        # A_tilde = jnp.block([[jnp.eye(self.Dz), -self.A.T]])
        T, Qz = self._reduce_batch_dims(stats)
        #return 0.5 * (Qz + Qz.T) / T
        return Qz / T

    def _get_Qz_stats(self, two_step_smooth_dict: dict):
        T = two_step_smooth_dict["mu"].shape[0]
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        A_tilde = jnp.eye(2 * self.Dz, self.Dz)
        A_tilde = A_tilde.at[self.Dz :].set(-self.A.T)
        b_tilde = -self.b
        Qz = jnp.sum(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
                A_mat=A_tilde.T,
                a_vec=b_tilde,
                B_mat=A_tilde.T,
                b_vec=b_tilde,
            ),
            axis=0,
        )
        return jnp.array([T]), Qz

    def update_state_density(self):
        """Update the state density."""
        Sigma_z = self.Qz + self.delta * jnp.eye(self.Dz)
        self.state_density = conditional.ConditionalGaussianPDF(
            M=jnp.array([self.A]), b=jnp.array([self.b]), Sigma=jnp.array([Sigma_z])
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def get_params(self) -> dict:
        return {
            "A": self.A,
            "b": self.b,
            "Lz": self.Lz,
        }

    @classmethod
    def from_dict(cls, params: dict):
        Dz = params["A"].shape[0]
        model = cls(Dz=Dz)
        model.A = params["A"]
        model.b = params["b"]
        model.Lz = params["Lz"]
        model.Qz = model.cholvec_to_mat(model.Lz, model.Dz)
        model.update_state_density()
        return model


class NNControlStateModel(LinearStateModel):
    r"""Model with linear state equation

    .. math::

        Z_t = A(u_t) Z_{t-1} + b(u_t) + \zeta_t \text{ with } \zeta_t \sim {\cal N}(0,\Sigma_z),

    where the coefficients are output of a neural network, which gets control variables u as input.

    :param Dz: Dimension of latent space
    :type Dz: int
    :param Du: Dimension of control variables.
    :type Du: int
    :param noise_z: Initial state noise, defaults to 1
    :type noise_z: float, optional
    :param hidden_units: List of number of hidden units in each layer, defaults to [16,]
    :type hidden_units: list, optional
    :param non_linearity: Which non-linearity between layers, defaults to objax.functional.tanh
    :type non_linearity: callable, optional
    :param lr: Learning rate for learning the network, defaults to 1e-4
    :type lr: float, optional
    """

    def __init__(
        self,
        Dz: int,
        Du: int,
        control_func: hk.Module = None,
        noise_z: float = 1,
        lr: float = 1e-4,
    ):
        self.Dz = Dz
        self.Qz = noise_z**2 * jnp.eye(self.Dz)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        self.Du = Du
        self.control_func_hk = self._setup_control_func(control_func)
        dummy_input = jnp.ones([1, Du])
        rng_key = random.PRNGKey(42)
        self.net_params = self.control_func_hk.init(rng_key, dummy_input)
        callable_control_func = lambda x: self.control_func_hk.apply(self.net_params, x)
        self.state_density = conditional.NNControlGaussianConditional(
            Sigma=jnp.array([self.Qz]),
            num_cond_dim=self.Dz,
            num_control_dim=self.Du,
            control_func=callable_control_func,
        )
        self.lr = lr

    def _setup_control_func(self, control_func: callable) -> hk.Module:
        if control_func == None:
            control_func = hk.transform(
                lambda x: hk.nets.MLP(
                    [10, self.Dz * (self.Dz + 1)],
                    activation=jax.nn.tanh,
                    w_init=jnp.zeros,
                )(x)
            )
        control_func = hk.without_apply_rng(control_func)
        return control_func

    def update_hyperparameters(
        self,
        smooth_dict: dict,
        two_step_smooth_dict: dict,
        control_z: jnp.ndarray,
        **kwargs
    ):
        """Update hyperparameters.

        The densities :math:`p(Z_t|X_{1:T})` and :math:`p(Z_{t+1}, Z_t|X_{1:T})` need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        """
        self._update_network_params(
            smooth_dict, two_step_smooth_dict, control_z, **kwargs
        )
        self.Qz = self._update_Qz(two_step_smooth_dict, control_z, **kwargs)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        self.update_state_density()

    def update_state_density(self):
        """Update the state density."""
        callable_control_func = lambda x: self.control_func_hk.apply(self.net_params, x)
        self.state_density.control_func = callable_control_func
        self.state_density.update_Sigma(jnp.array([self.Qz]))

    def _update_Qz(
        self, two_step_smooth_dict: pdf.GaussianPDF, control_z: jnp.ndarray, **kwargs
    ):
        """Update the transition covariance.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        """
        stats = jit(vmap(self._get_Qz_stats))(two_step_smooth_dict, control_z)
        T, Qz = self._reduce_batch_dims(stats)
        return Qz / T

    def _get_Qz_stats(
        self,
        two_step_smooth_dict: pdf.GaussianPDF,
        control_z: jnp.ndarray,
    ):
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        T = two_step_smoothing_density.R
        A_u, b_u = self.state_density.get_M_b(control_z)
        A_tilde = jnp.empty((two_step_smoothing_density.R, self.Dz, 2 * self.Dz))
        A_tilde = A_tilde.at[:, :, : self.Dz].set(jnp.eye(self.Dz))
        A_tilde = A_tilde.at[:, :, self.Dz :].set(-A_u)
        b_tilde = -b_u
        Qz = jnp.sum(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
                A_mat=A_tilde,
                a_vec=b_tilde,
                B_mat=A_tilde,
                b_vec=b_tilde,
            ),
            axis=0,
        )
        return jnp.array([T]), Qz

    def _update_network_params(
        self,
        smooth_dict: dict,
        two_step_smooth_dict: dict,
        control_z: jnp.ndarray,
        **kwargs
    ):
        """Update the network parameters by gradient descent.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        :param u: Control variables. Dimensions should be [T, Du]
        :type u: jnp.ndarray
        """

        def objective(params, smooth_dict, two_step_smooth_dict, control_z) -> float:
            smoothing_density = pdf.GaussianPDF(**smooth_dict)
            two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
            self.state_density.control_func = lambda x: self.control_func_hk.apply(
                params, x
            )
            return -self.compute_Q_function(
                smoothing_density, two_step_smoothing_density, control_z
            )

        batch_objective = (
            lambda params, smooth_dict, two_step_smooth_dict, control_z: jnp.mean(
                vmap(
                    objective,
                    in_axes=[
                        None,
                        {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0},
                        {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0},
                        0,
                    ],
                )(params, smooth_dict, two_step_smooth_dict, control_z)
            )
        )
        params = self.net_params
        result = minimize(
            batch_objective,
            params,
            "L-BFGS-B",
            args=(smooth_dict, two_step_smooth_dict, control_z),
        )
        self.net_params = result.x

    def compute_Q_function(
        self,
        smoothing_density: pdf.GaussianPDF,
        two_step_smoothing_density: pdf.GaussianPDF,
        control_z: jnp.ndarray,
        **kwargs
    ) -> float:
        r"""Compute state part of the Q-function.

        .. math::

            \sum \mathbb{E}[\ln p(Z_t|Z_{t-1})],

        where the expection is over the smoothing density.

        :param smoothing_density: Smoothing density :math:`p(Z_t|X_{1:T})`
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density  :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        :return: Evaluated state part of the Q-function.
        :rtype: float
        """
        state_density = self.state_density.set_control_variable(control_z)
        return jnp.sum(
            state_density.integrate_log_conditional(
                two_step_smoothing_density,
                p_x=smoothing_density.slice(jnp.arange(0, smoothing_density.R-1)),
            )
        )

    def get_params(self) -> dict:
        raise NotImplementedError("Not implemented yet.")

    @classmethod
    def from_dict(cls, params: dict):
        raise NotImplementedError("Not implemented yet.")


class LSEMStateModel(LinearStateModel):
    r"""This implements a linear+squared exponential mean (LSEM) state model

        .. math::

            Z_t = A \phi(Z_{t-1}) + b + \zeta_t  \text{ with } \zeta_t \sim {\cal N}(0,\Sigma_z).

        The feature function is

        .. math::

            \phi(Z) = (Z_0, Z_1,...,Z_m, k(h_1(Z))),...,k(h_n(Z)))^\top.

        The kernel and linear activation function are given by

        .. math::

            k(h) = \exp(-h^2 / 2) \text{ and } h_i(Z) = w_i^\top Z + w_{i,0}.

    :param Dz: Dimensionality of latent space.
    :type Dz: int
    :param Dk: Number of kernels to use.
    :type Dk: int
    :param noise_z: Initial isoptropic std. on the state transition., defaults to 1.0
    :type noise_z: float, optional
    """

    def __init__(
        self,
        Dz: int,
        Dk: int,
        noise_z: float = 1.0,
        lambda_W: float = 0.0,
        delta: float = 0.0,
        key=random.PRNGKey(42),
    ):
        self.Dz, self.Dk = Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qz = noise_z**2 * jnp.eye(self.Dz)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        self.lambda_W = lambda_W
        key, subkey = random.split(key)
        self.A = random.normal(subkey, (self.Dz, self.Dphi))
        self.A = self.A.at[:, : self.Dz].set(jnp.eye(self.Dz))
        self.b = jnp.zeros((self.Dz,))
        key, subkey = random.split(key)
        self.W = random.normal(subkey, (self.Dk, self.Dz + 1))
        self.delta = delta
        self.update_state_density()

    def update_hyperparameters(
        self, smooth_dict: dict, two_step_smooth_dict: dict, **kwargs
    ):
        """Update hyperparameters.

        The densities :math:`p(Z_t|X_{1:T})` and :math:`p(Z_{t+1}, Z_t|X_{1:T})` need to be provided (the latter for the cross-terms.)

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        self.A = jit(self._update_A)(smooth_dict, two_step_smooth_dict)
        self.Qz = jit(self._update_Qz)(smooth_dict, two_step_smooth_dict)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        self.b = jit(self._update_b)(smooth_dict)
        self.update_state_density()
        self._update_kernel_params(smooth_dict, two_step_smooth_dict)
        self.update_state_density()

    def _update_A(self, smooth_dict: dict, two_step_smooth_dict: dict):
        stats = jit(vmap(self._get_A_stats))(smooth_dict, two_step_smooth_dict)
        T, Eff, Ezf, Ef = self._reduce_batch_dims(stats)
        Ebf = (Ef[None] * self.b[:, None])
        # To prevent numerical issues
        Eff = Eff.at[self.Dz:,self.Dz:].add(1e-4 * jnp.eye(self.Dk))
        A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        # b = sd_mu / T - jnp.dot(A, Ef / T).T
        return A
    
    def _update_b(self, smooth_dict: dict):
        stats = jit(vmap(self._get_b_stats))(smooth_dict)
        T, Ef, sd_mu = self._reduce_batch_dims(stats)
        #Ebf = (Ef[None] * self.b[:, None]) / T
        # A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        b = sd_mu / T - jnp.dot(self.A, Ef / T).T
        return b
    
    def _update_Ab(self, smooth_dict: dict, two_step_smooth_dict: dict):
        stats = jit(vmap(self._get_Ab_stats))(smooth_dict, two_step_smooth_dict)
        T, Eff, Ezf, Ef, sd_mu = self._reduce_batch_dims(stats)
        Ebf = (Ef[None] * self.b[:, None]) / T
        A = jnp.linalg.solve(Eff / T, (Ezf - Ebf).T / T).T
        b = sd_mu / T - jnp.dot(A, Ef / T).T
        return A, b
    
    def _get_A_stats(self, smooth_dict: dict, two_step_smooth_dict: dict):
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        T = two_step_smoothing_density.R
        smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(T))
        joint_k_func = self._get_joint_k_function()
        two_step_k_measure = two_step_smoothing_density.multiply(
            joint_k_func, update_full=True
        )
        Ekz = jnp.sum(
            two_step_k_measure.integrate("x").reshape((T, self.Dk, 2 * self.Dz)), axis=0
        )
        Ekz_future, Ekz_past = Ekz[:, : self.Dz], Ekz[:, self.Dz :]
        sd_k = smoothing_density.multiply(self.state_density.k_func, update_full=True)
        sd_kk = sd_k.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.sum(sd_k.integral_light().reshape((T, self.Dk)), axis=0)
        Ez = jnp.sum(smoothing_density.integrate("x"), axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        # E[z f(z)']
        Ezz_cross = jnp.sum(
            two_step_smoothing_density.integrate("xx'")[:, self.Dz :, : self.Dz], axis=0
        )
        Ezf = jnp.concatenate([Ezz_cross.T, Ekz_future.T], axis=1)
        Ekk = sd_kk.integral_light().reshape((T, self.Dk, self.Dk))
        Ekz = sd_k.integrate("x").reshape((T, self.Dk, self.Dz))
        sum_Ekz = jnp.sum(Ekz, axis=0)
        sum_Ezz = jnp.sum(smoothing_density.integrate("xx'"), axis=0)
        sum_Ekk = jnp.sum(Ekk, axis=0)
        Eff = jnp.block([[sum_Ezz, sum_Ekz.T], [sum_Ekz, sum_Ekk]])
        #Eff += 0.001 * jnp.eye(Eff.shape[0])
        return jnp.array([T]), Eff, Ezf, Ef

    def _get_b_stats(self, smooth_dict: dict):
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        T = smoothing_density.R - 1
        sd_k = smoothing_density.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.sum(sd_k.integral_light().reshape((T+1, self.Dk))[:-1], axis=0)
        Ez = jnp.sum(smoothing_density.integrate("x")[:-1], axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        #Eff += 0.001 * jnp.eye(Eff.shape[0])
        sd_mu = jnp.sum(smoothing_density.mu[1:], axis=0)
        return jnp.array([T]), Ef, sd_mu
    
    def _get_Ab_stats(self, smooth_dict: dict, two_step_smooth_dict: dict):
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        T = two_step_smoothing_density.R
        smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(T))
        joint_k_func = self._get_joint_k_function()
        two_step_k_measure = two_step_smoothing_density.multiply(
            joint_k_func, update_full=True
        )
        Ekz = jnp.sum(
            two_step_k_measure.integrate("x").reshape((T, self.Dk, 2 * self.Dz)), axis=0
        )
        Ekz_future, Ekz_past = Ekz[:, : self.Dz], Ekz[:, self.Dz :]
        sd_k = smoothing_density.multiply(self.state_density.k_func, update_full=True)
        sd_kk = sd_k.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.sum(sd_k.integral_light().reshape((T, self.Dk)), axis=0)
        Ez = jnp.sum(smoothing_density.integrate("x")[:-1], axis=0)
        Ef = jnp.concatenate([Ez, Ek])
        # E[z f(z)']
        Ezz_cross = jnp.sum(
            two_step_smoothing_density.integrate("xx'")[:, self.Dz :, : self.Dz], axis=0
        )
        Ezf = jnp.concatenate([Ezz_cross.T, Ekz_future.T], axis=1)
        Ekk = sd_kk.integral_light().reshape((T, self.Dk, self.Dk))
        Ekz = sd_k.integrate("x").reshape((T, self.Dk, self.Dz))
        sum_Ekz = jnp.sum(Ekz, axis=0)
        sum_Ezz = jnp.sum(smoothing_density.integrate("xx'"), axis=0)
        sum_Ekk = jnp.sum(Ekk, axis=0)
        Eff = jnp.block([[sum_Ezz, sum_Ekz.T], [sum_Ekz, sum_Ekk]])
        #Eff += 0.001 * jnp.eye(Eff.shape[0])
        sd_mu = jnp.sum(smoothing_density.mu[1:], axis=0)
        return jnp.array([T]), Eff, Ezf, Ef, sd_mu

    def _update_Qz(self, smooth_dict: dict, two_step_smooth_dict: dict, **kwargs):
        """Update transition covariance.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`.
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """
        stats = vmap(self._get_Qz_stats)(smooth_dict, two_step_smooth_dict)
        T, Qz = self._reduce_batch_dims(stats)
        #return 0.5 * (Qz + Qz.T) / T
        return Qz / T
    
    def _get_Qz_stats(self, smooth_dict: dict, two_step_smooth_dict: dict):
        two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
        T = two_step_smoothing_density.R
        smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(T))
        A_tilde = jnp.block([[jnp.eye(self.Dz, self.Dz)], [-self.A[:, : self.Dz].T]])
        b_tilde = -self.b
        Qz_lin = jnp.sum(
            two_step_smoothing_density.integrate(
                "(Ax+a)(Bx+b)'",
                A_mat=A_tilde.T,
                a_vec=b_tilde,
                B_mat=A_tilde.T,
                b_vec=b_tilde,
            ),
            axis=0,
        )
        joint_k_func = self._get_joint_k_function()
        two_step_k_measure = two_step_smoothing_density.multiply(
            joint_k_func, update_full=True
        )
        Ekz = jnp.sum(
            two_step_k_measure.integrate("x").reshape((T, self.Dk, 2 * self.Dz)), axis=0
        )
        Ekz_future, Ekz_past = Ekz[:, : self.Dz], Ekz[:, self.Dz :]
        sd_k = smoothing_density.multiply(self.state_density.k_func, update_full=True)
        Ek = jnp.sum(sd_k.integral_light().reshape((T, self.Dk)), axis=0)
        Qz_k_lin_err = jnp.dot(
            self.A[:, self.Dz :],
            (
                Ekz_future
                - jnp.dot(self.A[:, : self.Dz], Ekz_past.T).T
                - Ek[:, None] * self.b[None]
            ),
        )
        sd_kk = sd_k.multiply(self.state_density.k_func, update_full=True)
        Ekk = jnp.sum(
            sd_kk.integral_light().reshape((T, self.Dk, self.Dk)),
            axis=0,
        )
        Qz_kk = jnp.dot(jnp.dot(self.A[:, self.Dz :], Ekk), self.A[:, self.Dz :].T)
        Qz = Qz_lin + Qz_kk - Qz_k_lin_err - Qz_k_lin_err.T
        # Qz_arr = 0.5 * (Qz + Qz.T)
        return jnp.array([T]), Qz

    def _get_joint_k_function(self):
        zero_arr = jnp.zeros([self.Dk, 2 * self.Dz])
        v_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.v)
        nu_joint = zero_arr.at[:, self.Dz :].set(self.state_density.k_func.nu)
        joint_k_func = factor.OneRankFactor(
            v=v_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta
        )
        return joint_k_func

    def update_state_density(self):
        """Update the state density."""
        Sigma_z = self.Qz + self.delta * jnp.eye(self.Dz)
        self.state_density = approximate_conditional.LSEMGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            W=self.W,
            Sigma=jnp.array([Sigma_z]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def _update_kernel_params(
        self,
        smooth_dict: dict,
        two_step_smooth_dict: dict,
    ):
        """Update the kernel weights.

        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """

        def objective(W, smooth_dict: dict, two_step_smooth_dict: dict):
            smoothing_density = pdf.GaussianPDF(**smooth_dict)
            two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
            self.state_density.w0 = W[:, 0]
            self.state_density.W = W[:, 1:]
            self.state_density.update_phi()
            return -self.compute_Q_function(
                smoothing_density, two_step_smoothing_density
            ) + 0.5 * self.lambda_W * jnp.sum(W**2)

        batch_objective = lambda params, smooth_dict, two_step_smooth_dict: jnp.mean(
            vmap(
                objective,
                in_axes=[
                    None,
                    {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0},
                    {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0},
                ],
            )(params, smooth_dict, two_step_smooth_dict)
        )
        params = self.W
        result = minimize(
            batch_objective,
            params,
            "L-BFGS-B",
            args=(smooth_dict, two_step_smooth_dict),
        )
        self.W = result.x

    def get_params(self) -> dict:
        return {
            "A": self.A,
            "b": self.b,
            "W": self.W,
            "Lz": self.Lz,
        }

    @classmethod
    def from_dict(cls, params: dict):
        Dz = params["A"].shape[0]
        Dk = params["W"].shape[0]
        model = cls(Dz=Dz, Dk=Dk)
        model.A = params["A"]
        model.b = params["b"]
        model.W = params["W"]
        model.Lz = params["Lz"]
        model.Qz = model.cholvec_to_mat(model.Lz, model.Dz)
        model.update_state_density()
        return model


class LRBFMStateModel(LSEMStateModel):
    r"""This implements a linear+RBF mean (LRBFM) state model

    .. math::

        Z_t = A \phi(Z_{t-1}) + b + \zeta_t  \text{ with } \zeta_t \sim N(0,\Sigma_z).

    The feature function is

    .. math::

        \phi(Z) = (Z_0, Z_1,...,Z_m, k(h_1(Z))),...,k(h_n(Z)))^\top.

    The kernel and linear activation function are given by

    .. math::

        k(h) = \exp(-\sum_i h_i^2 / 2) \text{ and } h_i(Z) = (Z_i - \mu_i) / l_i.

    :param Dz: Dimensionality of latent space.
    :type Dz: int
    :param Dk: Number of kernels to use.
    :type Dk: int
    :param noise_z: Initial isoptropic std. on the state transition., defaults to 1.0
    :type noise_z: float, optional
    :param kernel_type: Parameter determining, which kernel is used. 'scalar' same length scale for all kernels and
        dimensions. 'isotropic' same length scale for dimensions, but different for each kernel. 'anisotropic'
        different length scale for all kernels and dimensions., defaults to 'isotropic
    :type kernel_type: str
    """

    def __init__(
        self,
        Dz: int,
        Dk: int,
        noise_z: float = 1.0,
        kernel_type: bool = "isotropic",
        key=random.PRNGKey(42),
    ):
        self.Dz, self.Dk = Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qz = noise_z**2 * jnp.eye(self.Dz)
        self.Lz = self.from_mat_to_cholvec(self.Qz)
        key, subkey = random.split(key)
        self.A = random.normal(subkey, (self.Dz, self.Dphi))
        self.A = self.A.at[:, : self.Dz].set(jnp.eye(self.Dz))
        self.b = jnp.zeros((self.Dz,))
        key, subkey = random.split(key)
        self.mu = random.normal(subkey, (self.Dk, self.Dz))
        self.kernel_type = kernel_type
        key, subkey = random.split(key)
        if self.kernel_type == "scalar":
            self.log_length_scale = random.normal(
                subkey,
                (
                    1,
                    1,
                ),
            )
        elif self.kernel_type == "isotropic":
            self.log_length_scale = random.normal(
                subkey,
                (
                    self.Dk,
                    1,
                ),
            )
        elif self.kernel_type == "anisotropic":
            self.log_length_scale = random.normal(subkey, (self.Dk, self.Dz))
        else:
            raise NotImplementedError("Kernel type not implemented.")

        self.state_density = approximate_conditional.LRBFGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            mu=self.mu,
            length_scale=self.length_scale,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    @property
    def length_scale(self):
        if self.kernel_type == "scalar":
            return jnp.tile(jnp.exp(self.log_length_scale), (self.Dk, self.Dz))
        elif self.kernel_type == "isotropic":
            return jnp.tile(jnp.exp(self.log_length_scale), (1, self.Dz))
        elif self.kernel_type == "anisotropic":
            return jnp.exp(self.log_length_scale)

    def update_state_density(self):
        """Update the state density."""
        self.state_density = approximate_conditional.LRBFGaussianConditional(
            M=jnp.array([self.A]),
            b=jnp.array([self.b]),
            mu=self.mu,
            length_scale=self.length_scale,
            Sigma=jnp.array([self.Qz]),
        )
        self.Qz_inv, self.ln_det_Qz = (
            self.state_density.Lambda[0],
            self.state_density.ln_det_Sigma[0],
        )

    def _get_joint_k_function(self):
        Lambda_joint = jnp.zeros((self.Dk, 2 * self.Dz, 2 * self.Dz))
        Lambda_joint = Lambda_joint.at[:, self.Dz :, self.Dz :].set(
            self.state_density.k_func.Lambda
        )
        nu_joint = jnp.zeros([self.Dk, 2 * self.Dz])
        nu_joint = nu_joint.at[:, self.Dz :].set(self.state_density.k_func.nu)
        joint_k_func = factor.ConjugateFactor(
            Lambda=Lambda_joint, nu=nu_joint, ln_beta=self.state_density.k_func.ln_beta
        )
        return joint_k_func

    def _update_kernel_params(
        self,
        smooth_dict: dict,
        two_step_smooth_dict: dict,
    ):
        """Update the kernel weights.

        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density :math:`p(Z_t|X_{1:T})`
        :type smoothing_density: pdf.GaussianPDF
        :param two_step_smoothing_density: The two point smoothing density :math:`p(Z_{t+1}, Z_t|X_{1:T})`.
        :type two_step_smoothing_density: pdf.GaussianPDF
        """

        def objective(params, smooth_dict: dict, two_step_smooth_dict: dict):
            smoothing_density = pdf.GaussianPDF(**smooth_dict)
            two_step_smoothing_density = pdf.GaussianPDF(**two_step_smooth_dict)
            self.state_density.mu = params["mu"]
            self.state_density.length_scale = jnp.exp(params["log_length_scale"])
            self.state_density.update_phi()
            return -self.compute_Q_function(
                smoothing_density, two_step_smoothing_density
            )

        batch_objective = lambda params, smooth_dict, two_step_smooth_dict: jnp.mean(
            vmap(
                objective,
                in_axes=[
                    None,
                    {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0},
                    {"Sigma": 0, "mu": 0, "Lambda": 0, "ln_det_Sigma": 0},
                ],
            )(params, smooth_dict, two_step_smooth_dict)
        )
        params = {"mu": self.mu, "log_length_scale": self.log_length_scale}
        result = minimize(
            batch_objective,
            params,
            "L-BFGS-B",
            args=(smooth_dict, two_step_smooth_dict),
        )
        self.mu = result.x["mu"]
        self.log_length_scale = result.x["log_length_scale"]

    def get_params(self) -> dict:
        return {
            "A": self.A,
            "b": self.b,
            "mu": self.mu,
            "log_length_scale": self.log_length_scale,
            "Lz": self.Lz,
        }

    @classmethod
    def from_dict(cls, params: dict):
        Dz = params["A"].shape[0]
        Dk = params["mu"].shape[0]
        model = cls(Dz=Dz, Dk=Dk)
        model.A = params["A"]
        model.b = params["b"]
        model.mu = params["mu"]
        model.log_length_scale = params["log_length_scale"]
        model.Lz = params["Lz"]
        model.Qz = model.cholvec_to_mat(model.Lz, model.Dz)
        model.update_state_density()
        return model
