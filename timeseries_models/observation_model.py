__author__ = "Christian Donner"
import scipy
from typing import Tuple
from scipy.optimize import minimize_scalar
from jax import numpy as jnp
from jax import scipy as jsc
import numpy as np
from jax import lax
from jax import jit, grad, vmap
from gaussian_toolbox import pdf, conditional, approximate_conditional
from gaussian_toolbox.utils.jax_minimize_wrapper import minimize
from jax import random

class ObservationModel:
    def __init__(self):
        """This is the template class for observation models in state space models.
        Basically these classes should contain all functionality for the mapping between
        the latent variables z, and observations x, i.e. p(x_t|z_t). The object should
        have an attribute `observation_density`, which is be a `ConditionalDensity`.
        Furthermore, it should be possible to optimize hyperparameters, when provided
        with a density over the latent space.
        """
        self.observation_density = None

    def filtering(
        self, prediction_density: pdf.GaussianPDF, x_t: jnp.ndarray, **kwargs
    ) -> pdf.GaussianPDF:
        """Calculate filter density.

        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)

        :param prediction_density: Prediction density p(z_t|x_{1:t-1}).
        :type prediction_density: pdf.GaussianPDF
        :param x_t: Observation vector. Dimensions should be [1, Dx].
        :type x_t: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        :return: Filter density p(z_t|x_{1:t}).
        :rtype: pdf.GaussianPDF
        """
        raise NotImplementedError("Filtering for observation model not implemented.")

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
        

class LinearObservationModel(ObservationModel):
    def __init__(self, Dx: int, Dz: int, noise_x: float = 1.0):
        """This class implements a linear observation model, where the observations are generated as

            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx).

        :param Dx: Dimensionality of observations.
        :type Dx: int
        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param noise_x: Intial isoptropic std. on the observations., defaults to 1.0
        :type noise_x: float, optional
        """
        self.Dx, self.Dz = Dx, Dz
        if Dx == Dz:
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
        self.d = jnp.zeros(Dx)
        self.Qx = noise_x**2 * jnp.eye(self.Dx)
        self.observation_density = conditional.ConditionalGaussianPDF(
            M=jnp.array([self.C]), b=jnp.array([self.d]), Sigma=jnp.array([self.Qx])
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )

    def filtering(
        self, prediction_density: pdf.GaussianPDF, x_t: jnp.ndarray, **kwargs
    ) -> pdf.GaussianPDF:
        """_"Calculate filter density.

        p(z_t|x_{1:t}) = p(x_t|z_t)p(z_t|x_{1:t-1}) / p(x_t)

        :param prediction_density: Prediction density p(z_t|x_{1:t-1}).
        :type prediction_density: pdf.GaussianPDF
        :param x_t: Observation vector. Dimensions should be [1, Dx].
        :type x_t: jnp.ndarray
        :return: Filter density p(z_t|x_{1:t}).
        :rtype: pdf.GaussianPDF
        """
        # p(z_t| x_t, x_{1:t-1})
        p_z_given_x = self.observation_density.affine_conditional_transformation(
            prediction_density
        )
        # Condition on x_t
        cur_filter_density = p_z_given_x.condition_on_x(x_t)
        return cur_filter_density

    def partially_observed_filtering(
        self,
        prediction_density: pdf.GaussianPDF,
        x_t: jnp.ndarray,
        observed_dims: jnp.ndarray = None,
        **kwargs
    ) -> pdf.GaussianPDF:
        """Calculate the filtering density for incomplete data, i.e. some fixed dimension are not observed.

        :param prediction_density: Prediction density p(z_t|x_{1:t-1}).
        :type prediction_density: pdf.GaussianPDF
        :param x_t: Observation vector. Dimensions should be [1, Dx]. Not observed values should be nans.
        :type x_t: jnp.ndarray
        :param observed_dims: Dimensions that are observed. If non empty set., defaults to None
        :type observed_dims: jnp.ndarray, optional
        :return: Filter density p(z_t|x_{1:t}).
        :rtype: pdf.GaussianPDF
        """
        # In case all data are unobserved
        if observed_dims == None:
            return prediction_density
        # In case all data are observed
        elif len(observed_dims) == self.Dx:
            cur_filter_density = self.filtering(prediction_density, x_t)
            return cur_filter_density
        # In case we have only partial observations
        else:
            # p(z_t, x_t| x_{1:t-1})
            p_zx = self.observation_density.affine_joint_transformation(
                prediction_density
            )
            # p(z_t, x_t (observed) | x_{1:t-1})
            marginal_dims = jnp.concatenate(
                [jnp.arange(self.Dz), self.Dz + observed_dims]
            )
            p_zx_observed = p_zx.get_marginal(marginal_dims)
            # p(z_t | x_t (observed), x_{1:t-1})
            conditional_dims = jnp.arange(self.Dz, self.Dz + len(observed_dims))
            nonconditional_dims = jnp.arange(0, self.Dz)
            p_z_given_x_observed = p_zx_observed.condition_on_explicit(
                conditional_dims, nonconditional_dims
            )
            cur_filter_density = p_z_given_x_observed.condition_on_x(
                x_t[:, observed_dims]
            )

            return cur_filter_density

    def compute_Q_function(
        self, smoothing_density: pdf.GaussianPDF, X: jnp.ndarray, **kwargs
    ) -> float:
        return jnp.sum(
            self.observation_density.integrate_log_conditional_y(smoothing_density, y=X)
        )

    def update_hyperparameters(
        self, X: jnp.ndarray, smooth_dict: dict, **kwargs
    ):
        """Update hyperparameters.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        self.C = jit(self._update_C)(X, smooth_dict)
        self.d = jit(self._update_d)(X, smooth_dict)
        self.Qx = jit(self._update_Qx)(X, smooth_dict)
        self.update_observation_density()
        
    def _update_C(self, X, smooth_dict: pdf.GaussianPDF):
        """Update observation matrix.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        stats = vmap(self._get_C_stats)(X, smooth_dict)
        A, b = self._reduce_batch_dims(stats)
        C_new = jnp.linalg.solve(A, b).T
        return C_new
        
    def _get_C_stats(self, X: jnp.ndarray, smooth_dict: dict):
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        Ezz = jnp.sum(smoothing_density.integrate("xx'")[1:], axis=0)
        Ez = smoothing_density.integrate("x")[1:]
        zx = jnp.sum(Ez[:, :, None] * (X[:, None] - self.d[None, None]), axis=0)
        return Ezz, zx

    def _update_Qx(self, X: jnp.ndarray, smooth_dict: dict):
        """Update observation covariance matrix.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        stats = vmap(self._get_Qx_stats)(X, smooth_dict)
        T, Qx = self._reduce_batch_dims(stats)
        Qx = 0.5 * (Qx + Qx.T) / T
        return Qx
        
    def _get_Qx_stats(self, X, smooth_dict):
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        T = X.shape[0]
        A = -self.C
        a_t = jnp.concatenate([self.d[None], X]) - self.d[None]
        Qx = jnp.sum(
            smoothing_density.integrate(
                "(Ax+a)(Bx+b)'", A_mat=A, a_vec=a_t, B_mat=A, b_vec=a_t
            )[1:],
            axis=0,
        )
        return jnp.array([T]), Qx

    def _update_d(self,X: jnp.ndarray, smooth_dict: dict):
        """Update observation offset.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        stats = vmap(self._get_d_stats)(X, smooth_dict)
        T, diff_d = self._reduce_batch_dims(stats)
        d = diff_d / T
        return d
        
    def _get_d_stats(self, X: jnp.array, smooth_dict):
        smoothing_density = pdf.GaussianPDF(**smooth_dict)
        T = X.shape[0]
        diff_d = jnp.sum(X - jnp.dot(smoothing_density.mu[1:], self.C.T), axis=0)
        return jnp.array([T]), diff_d

    def update_observation_density(self):
        """Updates the emission density."""
        self.observation_density = conditional.ConditionalGaussianPDF(
            M=jnp.array([self.C]), b=jnp.array([self.d]), Sigma=jnp.array([self.Qx])
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )
        
    def get_data_density(self, p_z: pdf.GaussianPDF, **kwargs):
        return self.observation_density.affine_marginal_transformation(p_z)

    def condition_on_z_and_observations(
        self,
        z_sample: jnp.ndarray,
        x_t: jnp.ndarray,
        observed_dims: jnp.ndarray,
        unobserved_dims: jnp.ndarray,
        **kwargs
    ) -> pdf.GaussianPDF:
        """Returns the density p(x_unobserved|X_observed=x, Z=z).

        :param z_sample: Values of latent variable
        :type z_sample: jnp.ndarray
        :param x_t: Data.
        :type x_t: jnp.ndarray
        :param observed_dims: Observed dimension
        :type observed_dims: jnp.ndarray
        :param unobserved_dims: Unobserved dimensions.
        :type unobserved_dims: jnp.ndarray
        :return: The density over unobserved dimensions.
        :rtype: pdf.GaussianPDF
        """
        p_x = self.observation_density.condition_on_x(z_sample)
        if observed_dims != None:
            p_x = p_x.condition_on_explicit(observed_dims, unobserved_dims)
            p_x = p_x.condition_on_x(x_t[observed_dims][None])
        return p_x

class LSEMObservationModel(LinearObservationModel):
    def __init__(
        self,
        Dx: int,
        Dz: int,
        Dk: int,
        noise_x: float = 1.0,
    ):
        """
        This implements a linear+squared exponential mean (LSEM) observation model

            x_t = C phi(z_{t}) + d + xi_t     with      xi_t ~ N(0,Qx).

            The feature function is

            phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).

            The kernel and linear activation function are given by

            k(h) = exp(-h^2 / 2) and h_i(x) = w_i'x + w_{i,0}.


        :param Dx: Dimensions of observations.
        :type Dx: int
        :param Dz: Dimensions of latent space.
        :type Dz: int
        :param Dk: Number of kernels.
        :type Dk: int
        :param noise_x: Initial observation noise, defaults to 1.0
        :type noise_x: float, optional
        """
        self.Dx, self.Dz, self.Dk = Dx, Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qx = noise_x**2 * jnp.eye(self.Dx)
        self.C = jnp.array(np.random.randn(self.Dx, self.Dphi))
        if self.Dx == self.Dz:
            self.C = self.C.at[:, : self.Dz].set(jnp.eye(self.Dx))
        else:
            self.C = self.C.at[:, : self.Dz].set(
                jnp.array(np.random.randn(self.Dx, self.Dz))
            )
        self.d = jnp.zeros((self.Dx,))
        self.W = jnp.array(np.random.randn(self.Dk, self.Dz + 1))
        self.observation_density = approximate_conditional.LSEMGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            W=self.W,
            Sigma=jnp.array([self.Qx]),
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )

    def update_hyperparameters(
        self, X: jnp.ndarray, smooth_dict: dict, **kwargs
    ):
        """Update the hyperparameters C,d,Qx,W.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations.
        :type X: jnp.ndarray
        """
        self.Qx = jit(self._update_Qx)(X, smooth_dict)
        self.C, self.d = jit(self._update_Cd)(X, smooth_dict)
        self.update_observation_density()
        self._update_kernel_params(X, smooth_dict)
        self.update_observation_density()

    def update_observation_density(self):
        """Create new emission density with current parameters."""
        self.observation_density = approximate_conditional.LSEMGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            W=self.W,
            Sigma=jnp.array([self.Qx]),
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )
        
    def _get_Qx_stats(self, X, smooth_dict):
        T = X.shape[0]
        smoothing_density = pdf.GaussianPDF(smooth_dict).slice(jnp.arange(1,T+1))
        mu_x, Sigma_x = self.observation_density.get_expected_moments(smoothing_density)
        sum_mu_x2 = jnp.sum(
            Sigma_x - self.observation_density.Sigma + mu_x[:, None] * mu_x[:, :, None],
            axis=0,
        )
        sum_X_mu = jnp.sum(X[:, None] * mu_x[:, :, None], axis=0)
        Qx = jnp.sum(X[:, None] * X[:, :, None], axis=0) - sum_X_mu - sum_X_mu.T + sum_mu_x2
        return jnp.array([T]), Qx

    def _update_Cd(self, X: jnp.array, smooth_dict: dict):
        """Update observation observation matrix C and vector d.

        C* = E[(X - d)phi(z)']E[phi(z)phi(z)']^{-1}
        d* = E[(X - C phi(x))]

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations.
        :type X: jnp.ndarray
        """
        stats = vmap(self._get_Cd_stats)(X, smooth_dict)
        T, A, B, Ef, sum_X = self._reduce_batch_dims(stats) 
        C = jnp.linalg.solve(A / T, B.T / T).T
        d = (sum_X - jnp.dot(C, Ef)) / T
        return C, d
        
    def _get_Cd_stats(self, X: jnp.ndarray, smooth_dict: dict):
        T = X.shape[0]
        smoothing_density = pdf.GaussianPDF(smooth_dict).slice(jnp.arange(1,T+1))
        Ex = smoothing_density.integrate("x")
        # E[k(x)] [R, Dphi - Dx]
        sd_k = smoothing_density.multiply(
            self.observation_density.k_func, update_full=True
        )
        Ekx = sd_k.integrate().reshape((smoothing_density.R, self.Dphi - self.Dz))
        # E[f(x)]
        Ef = jnp.concatenate([Ex, Ekx], axis=1)
        B = jnp.einsum("ab,ac->bc", X - self.d[None], Ef)
        sum_Ef = jnp.sum(Ef, axis=0)
        #### E[f(x)f(x)'] ####
        # Linear terms E[xx']
        Exx = jnp.sum(smoothing_density.integrate("xx'"), axis=0)
        # Cross terms E[x k(x)']
        Ekx = jnp.sum(
            sd_k.integrate("x").reshape((smoothing_density.R, self.Dk, self.Dz)), axis=0
        )
        # kernel terms E[k(x)k(x)']
        Ekk = jnp.sum(
            sd_k.multiply(self.observation_density.k_func, update_full=True)
            .integrate()
            .reshape((smoothing_density.R, self.Dk, self.Dk)),
            axis=0,
        )
        # Eff[:,self.Dx:,self.Dx:] = Ekk
        A = jnp.block([[Exx, Ekx.T], [Ekx, Ekk]])
        sum_X = jnp.sum(X, axis=0)
        return jnp.array([T]), A, B, sum_Ef, sum_X
        

    def _update_kernel_params(self, X: jnp.ndarray, smooth_dict: dict):
        """Update the kernel weights.

        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations.
        :type X: jnp.ndarray
        """

        def objective(W, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.w0 = W[:, 0]
            self.observation_density.W = W[:, 1:]
            self.observation_density.update_phi()
            return -self.compute_Q_function(smoothing_density, X)

        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        params = self.W
        result  = minimize(batch_objective, params, 'L-BFGS-B', args=(X, smooth_dict))
        self.W = result.x


class LRBFMObservationModel(LSEMObservationModel):
    def __init__(
        self,
        Dx: int,
        Dz: int,
        Dk: int,
        noise_z: float = 1.0,
        kernel_type: bool = "isotropic",
    ):
        """This implements a linear+RBF mean (LRBFM) observation model

            x_t = C phi(z_t) + d + xi_t     with      xi_t ~ N(0,Qx).

            The feature function is

            phi(x) = (x_0, x_1,...,x_m, k(h_1(x))),...,k(h_n(x))).

            The kernel and linear activation function are given by

            k(h) = exp(-h^2 / 2) and h_i(x) = (x_i + mu_i) / l_i.


        :param Dx: Dimensions of observations.
        :type Dx: int
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
        self.Dx, self.Dz, self.Dk = Dx, Dz, Dk
        self.Dphi = self.Dk + self.Dz
        self.Qx = noise_z**2 * jnp.eye(self.Dx)
        self.C = jnp.array(np.random.randn(self.Dx, self.Dphi))
        if self.Dx == self.Dz:
            self.C = self.C.at[:, : self.Dz].set(jnp.eye(self.Dx))
        else:
            self.C = self.C.at[:, : self.Dz].set(
                jnp.array(
                np.random.randn(self.Dx, self.Dz))
            )
        self.d = jnp.zeros((self.Dx,))
        self.mu = jnp.array(
                np.random.randn(self.Dk, self.Dz))
        self.kernel_type = kernel_type
        if self.kernel_type == "scalar":
            self.log_length_scale = jnp.array(
                np.random.randn(1, 1)
            )
        elif self.kernel_type == "isotropic":
            self.log_length_scale = jnp.array(
                np.random.randn(self.Dk, 1)
            )
        elif self.kernel_type == "anisotropic":
            self.log_length_scale = jnp.array(
                np.random.randn(self.Dk, self.Dz)
            )
        else:
            raise NotImplementedError("Kernel type not implemented.")

        self.observation_density = approximate_conditional.LRBFGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            mu=self.mu,
            length_scale=self.length_scale,
            Sigma=jnp.array([self.Qx]),
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )

    @property
    def length_scale(self):
        if self.kernel_type == "scalar":
            return jnp.tile(jnp.exp(self.log_length_scale), (self.Dk, self.Dz))
        elif self.kernel_type == "isotropic":
            return jnp.tile(jnp.exp(self.log_length_scale), (1, self.Dz))
        elif self.kernel_type == "anisotropic":
            return jnp.exp(self.log_length_scale)
        
    def update_observation_density(self):
        """Create new emission density with current parameters."""
        self.observation_density = approximate_conditional.LRBFGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            mu=self.mu,
            length_scale=self.length_scale,
            Sigma=jnp.array([self.Qx]),
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )
        

    def _update_kernel_params(self, X: jnp.ndarray, smooth_dict: dict):
        """Update the kernel weights.

        Using gradient descent on the (negative) Q-function.

        :param smoothing_density: The smoothing density  p(z_t|x_{1:T})
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations.
        :type X: jnp.ndarray
        """

        def objective(params, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.mu = params['mu']
            self.observation_density.length_scale = jnp.exp(params['log_length_scale'])
            self.observation_density.update_phi()
            return -self.compute_Q_function(smoothing_density, X)

        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        params = {'mu': self.mu, 'log_length_scale': self.log_length_scale}
        result  = minimize(batch_objective, params, 'L-BFGS-B', args=(X, smooth_dict))
        self.mu = result.x['mu']
        self.log_length_scale = result.x['log_length_scale']
        
'''        
class HCCovObservationModel(LinearObservationModel):
    def __init__(self, Dx: int, Dz: int, Du: int, noise_x: float = 1.0):
        r"""This class implements a linear observation model, where the observations are generated as

            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx(z_t)),

        where

            Qx(z) = sigma_x^2 I + \sum_i U_i D_i(z) U_i',

        with D_i(z) = 2 * beta_i * cosh(h_i(z)) and h_i(z) = w_i'z + b_i.

        :param Dx: Dimensionality of observations.
        :type Dx: int
        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param Du: Number of noise directions.
        :type Du: int
        :param noise_x: Intial isoptropic std. on the observations., defaults to 1.0
        :type noise_x: float, optional
        """
        self.update_noise = True
        assert Dx >= Du
        self.Dx, self.Dz, self.Du = Dx, Dz, Du
        if Dx == Dz:
            # self.C = jnp.eye(Dx)[:,:Dz]
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
            self.C = self.C / jnp.sqrt(jnp.sum(self.C**2, axis=0))[None]
        self.d = jnp.zeros(Dx)
        rand_mat = np.random.rand(self.Dx, self.Dx) - 0.5
        Q, R = np.linalg.qr(rand_mat)
        #self.U = jnp.array(Q[:, : self.Du])
        self.U = jnp.eye(self.Dx)[:,:self.Du]
        W = 1e-2 * np.random.randn(self.Du, self.Dz + 1)
        W[:, 0] = 0
        self.W = jnp.array(W)
        self.Sigma = .5 * noise_x ** 2 * jnp.eye(self.Dx)
        self.observation_density = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            Sigma=jnp.array([self.Sigma]),
            U=self.U,
            W=self.W,
        )

    def lin_om_init(self, lin_om: LinearObservationModel):
        # self.sigma_x = jnp.array([jnp.sqrt(jnp.amin(lin_om.Qx.diagonal()))])
        self.Sigma = .5 * lin_om.Qx
        _, U = np.linalg.eig(lin_om.Qx)
        _, U = jnp.array(U)
        if self.Dx != 1:
            self.U = jnp.real(U[:, : self.Du])
            # self.beta = jnp.abs(jnp.real(beta[:self.Du]))
            # self.beta.at[(self.beta / self.sigma_x ** 2) < .5].set(.5 / self.sigma_x ** 2)
        #self.C = lin_om.C
        #self.d = lin_om.d
        self.update_observation_density()


    def update_hyperparameters(
        self, X: jnp.ndarray, smooth_dict: dict, **kwargs
    ):
        """Update hyperparameters.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        #if self.update_noise:
        #    print('Update Sigma')
        self._update_Sigma(X, smooth_dict)
        #else:
        #    print('Update Rest')
        self._update_euclid_params(X, smooth_dict)
        if self.Dx > 1:
            self._update_stiefel_params(X, smooth_dict)
            
        self.update_noise = not self.update_noise
        print(self.W, self.Sigma)
        
    def _update_Sigma(self, X: jnp.ndarray, smooth_dict: dict):
        
        def get_Sigma_stats(X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            Sigma_stats = jnp.mean(self.observation_density.integrate_Sigma_stats(p_x=smoothing_density, y=X), axis=0)
            return Sigma_stats
        
        Sigma_stats_batch = vmap(get_Sigma_stats, in_axes=[0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(X, smooth_dict)
        Sigma_stats = jnp.mean(Sigma_stats_batch, axis=0)
        self.Sigma = Sigma_stats
        self.update_observation_density()
        
    def update_observation_density(self):
        """Updates the emission density."""
        self.observation_density = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            Sigma=jnp.array([self.Sigma]),
            U=self.U,
            W=self.W,
        )

    def _update_stiefel_params(self, X: jnp.ndarray, smooth_dict: dict, conv_crit: float=1e-4):
        
        def objective(U, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.U = U
            return -self.compute_Q_function(smoothing_density, X)
        
        U = self.U
        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        batch_objective_jit = jit(lambda U: batch_objective(U, X, smooth_dict))
        if self.Du > 1:
            def get_geodesic(U, objective):
                euclid_dU = grad(objective)(U)
                tangent = jnp.dot(U.T, euclid_dU) - jnp.dot(euclid_dU.T, U)
                geodesic = lambda t: jnp.dot(U, jsc.linalg.expm(t * tangent))
                return geodesic
            
            def geodesic_objective(t, geodesic, objective):
                return objective(geodesic(t))
            
            converged = False
            objective_val_old = 999999
            max_iter = 1000
            num_iter = 0
            while not converged and num_iter < max_iter:
                geodesic = get_geodesic(U, batch_objective_jit)
                geo_obj = lambda t: geodesic_objective(t, geodesic, batch_objective_jit)
                min_res = minimize_scalar(geo_obj, tol=conv_crit)
                opt_t = min_res.x
                U = geodesic(opt_t)
                objective_val_new = min_res.fun
                converged = self._check_convergence(objective_val_old, objective_val_new, conv_crit)
                objective_val_old = objective_val_new
                num_iter += 1
        else:
            def normed_objective(U_unnormed):
                U = U_unnormed / jnp.linalg.norm(U_unnormed, axis=0)
                return batch_objective_jit(U)
            
            result  = minimize(normed_objective, U, 'L-BFGS-B')
            U_unnormed = result.x
            U = U_unnormed / jnp.linalg.norm(U_unnormed, axis=0) 
        self.U = U
        self.update_observation_density()
            
            
    @staticmethod
    def _check_convergence(Q_old, Q_new: float, conv_crit: float) -> bool:
        conv = jnp.abs(Q_old - Q_new) / jnp.amax(
                    jnp.array(
                        [1, jnp.abs(Q_old), jnp.abs(Q_new)]
                    )
                )
        return conv < conv_crit
    
    def _update_euclid_params(self, X, smooth_dict):
        def objective(params, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.M = jnp.array([params['C']])
            self.observation_density.b = jnp.array([params['d']])
            self.observation_density.W = params['W']
            return -self.compute_Q_function(smoothing_density, X)

        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        params = {'C': self.C,
                  'd': self.d,
                  'W': self.W}
        result  = minimize(batch_objective, params, 'L-BFGS-B', args=(X, smooth_dict))
        self.C = result.x['C']
        self.d = result.x['d']
        self.W = result.x['W']
        self.update_observation_density()

'''       
class HCCovObservationModel(LinearObservationModel):
    def __init__(self, Dx: int, Dz: int, Du: int, noise_x: float = 1.0):
        r"""This class implements a linear observation model, where the observations are generated as

            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx(z_t)),

        where

            Qx(z) = sigma_x^2 I + \sum_i U_i D_i(z) U_i',

        with D_i(z) = 2 * beta_i * cosh(h_i(z)) and h_i(z) = w_i'z + b_i.

        :param Dx: Dimensionality of observations.
        :type Dx: int
        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param Du: Number of noise directions.
        :type Du: int
        :param noise_x: Intial isoptropic std. on the observations., defaults to 1.0
        :type noise_x: float, optional
        """
        self.update_noise = True
        assert Dx >= Du
        self.Dx, self.Dz, self.Du = Dx, Dz, Du
        if Dx == Dz:
            # self.C = jnp.eye(Dx)[:,:Dz]
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
            self.C = self.C / jnp.sqrt(jnp.sum(self.C**2, axis=0))[None]
        self.d = jnp.zeros(Dx)
        rand_mat = np.random.rand(self.Dx, self.Dx) - 0.5
        Q, R = np.linalg.qr(rand_mat)
        #self.U = jnp.array(Q[:, : self.Du])
        self.U = jnp.eye(self.Dx)[:,:self.Du]
        W = 1e-2 * np.random.randn(self.Du, self.Dz + 1)
        W[:, 0] = 0
        self.W = jnp.array(W)
        self.log_beta = jnp.log(2 * noise_x**2 * jnp.ones(self.Du))
        self.log_sigma_x = jnp.log(jnp.array([noise_x]))
        self.observation_density = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            sigma_x=self.sigma_x,
            U=self.U,
            W=self.W,
            beta=self.beta,
        )

    def lin_om_init(self, lin_om: LinearObservationModel):
        # self.sigma_x = jnp.array([jnp.sqrt(jnp.amin(lin_om.Qx.diagonal()))])
        beta, U = np.linalg.eig(lin_om.Qx)
        beta, U = jnp.array(beta), jnp.array(U)
        if self.Dx != 1:
            self.U = jnp.real(U[:, : self.Du])
            # self.beta = jnp.abs(jnp.real(beta[:self.Du]))
            # self.beta.at[(self.beta / self.sigma_x ** 2) < .5].set(.5 / self.sigma_x ** 2)
        self.C = lin_om.C
        self.d = lin_om.d
        self.update_observation_density()

    def pca_init(self, X: jnp.ndarray, smooth_window: int = 10):
        r"""Set the model parameters to an educated initial guess, based on principal component analysis.

        More specifically `d` is set to the mean of the data and `C` to the first principal components
        of the (smoothed) data.
        Then `U` is initialized with the first pricinpal components of the empirical covariance of the
        residuals.

        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :param smooth_window: Width of the box car filter data are smoothed with., defaults to 10
        :type smooth_window: int, optional
        """
        self.d = jnp.mean(X, axis=0)
        T = X.shape[0]
        X_smoothed = np.empty(X.shape)
        for i in range(X.shape[1]):
            X_smoothed[:, i] = np.convolve(
                X[:, i], np.ones(smooth_window) / smooth_window, mode="same"
            )
        eig_vals, eig_vecs = scipy.linalg.eigh(
            jnp.dot((X_smoothed - self.d[None]).T, X_smoothed - self.d[None]),
            eigvals=(self.Dx - np.amin([self.Dz, self.Dx]), self.Dx - 1),
        )
        C = np.array(self.C)
        C[:, : np.amin([self.Dz, self.Dx])] = eig_vecs * eig_vals / T
        self.C = jnp.array(self.C)
        z_hat = jnp.dot(jnp.linalg.pinv(self.C), (X_smoothed - self.d).T).T
        delta_X = X - jnp.dot(z_hat, self.C.T) - self.d
        cov = jnp.dot(delta_X.T, delta_X)
        self.U = jnp.array(
            scipy.linalg.eigh(cov, eigvals=(self.Dx - self.Du, self.Dx - 1))[1]
        )
        self.observation_density = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            sigma_x=self.sigma_x,
            U=self.U,
            W=self.W,
            beta=self.beta,
        )
        
    def compute_Q_function(
        self, smoothing_density: pdf.GaussianPDF, X: jnp.ndarray, z_samples=None, **kwargs
    ) -> float:
        return jnp.sum(
            self.observation_density.integrate_log_conditional_y(smoothing_density, y=X, x_samples=z_samples)
        )
        
    @property
    def beta(self):
        return jnp.exp(self.log_beta)
        
    @property
    def sigma_x(self):
        return jnp.exp(self.log_sigma_x)

    def update_hyperparameters(
        self, X: jnp.ndarray, smooth_dict: dict, **kwargs
    ):
        """Update hyperparameters.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
    
        #if self.update_noise:
        #    self._update_euclid_params2(X, smooth_dict)
        #else:
        self._update_euclid_params(X, smooth_dict)
        self.update_noise = not self.update_noise
        if self.Dx > 1:
            self._update_stiefel_params(X, smooth_dict)

    def update_observation_density(self):
        """Updates the emission density."""
        self.observation_density = approximate_conditional.HCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            sigma_x=self.sigma_x,
            U=self.U,
            W=self.W,
            beta=self.beta,
        )
        
    def sample_smoothing_density(self, X, smooth_dict):
        T = X.shape[0]
        key = random.PRNGKey(42)
        N = 1000
        smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
        z_samples = smoothing_density.sample(key, N)
        return z_samples
        

    def _update_stiefel_params(self, X: jnp.ndarray, smooth_dict: dict, conv_crit: float=1e-4):
        
        def objective(U, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.U = U
            return -self.compute_Q_function(smoothing_density, X)
        
        U = self.U
        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        batch_objective_jit = jit(lambda U: batch_objective(U, X, smooth_dict))
        if self.Du > 1:
            def get_geodesic(U, objective):
                euclid_dU = grad(objective)(U)
                tangent = jnp.dot(U.T, euclid_dU) - jnp.dot(euclid_dU.T, U)
                geodesic = lambda t: jnp.dot(U, jsc.linalg.expm(t * tangent))
                return geodesic
            
            def geodesic_objective(t, geodesic, objective):
                return objective(geodesic(t))
            
            converged = False
            objective_val_old = 999999
            max_iter = 1000
            num_iter = 0
            while not converged and num_iter < max_iter:
                geodesic = get_geodesic(U, batch_objective_jit)
                geo_obj = lambda t: geodesic_objective(t, geodesic, batch_objective_jit)
                # bounds for numerical stability
                min_res = minimize_scalar(geo_obj, tol=conv_crit, bounds=[-jnp.pi, jnp.pi])
                opt_t = min_res.x
                U = geodesic(opt_t)
                objective_val_new = min_res.fun
                converged = self._check_convergence(objective_val_old, objective_val_new, conv_crit)
                objective_val_old = objective_val_new
                num_iter += 1
        else:
            def normed_objective(U_unnormed):
                U = U_unnormed / jnp.linalg.norm(U_unnormed, axis=0)
                return batch_objective_jit(U)
            
            result  = minimize(normed_objective, U, 'L-BFGS-B')
            U_unnormed = result.x
            U = U_unnormed / jnp.linalg.norm(U_unnormed, axis=0) 
        self.U = U
        self.update_observation_density()
            
            
    @staticmethod
    def _check_convergence(Q_old, Q_new: float, conv_crit: float) -> bool:
        conv = jnp.abs(Q_old - Q_new) / jnp.amax(
                    jnp.array(
                        [1, jnp.abs(Q_old), jnp.abs(Q_new)]
                    )
                )
        return conv < conv_crit
    
    def _update_euclid_params(self, X, smooth_dict):

        def objective(params, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.M = jnp.array([params['C']])
            self.observation_density.b = jnp.array([params['d']])
            self.observation_density.W = params['W']
            self.observation_density.sigma_x = jnp.exp(params['log_sigma_x'])
            self.observation_density.sigma2_x = (jnp.exp(params['log_sigma_x'])) ** 2
            self.observation_density.beta = .25 * jnp.exp(2 * params['log_sigma_x']) + jnp.exp(params['y'])
            return -self.compute_Q_function(smoothing_density, X)
        


        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        params = {'log_sigma_x': jnp.log(jnp.exp(self.log_sigma_x)),
                  'y': jnp.log(self.beta - 0.25 * self.sigma_x**2),
                  'C': self.C,
                  'd': self.d,
                  'W': self.W}
        result  = minimize(batch_objective, params, 'L-BFGS-B', args=(X, smooth_dict))
        self.log_sigma_x = jnp.log(jnp.exp(result.x['log_sigma_x']))
        self.log_beta = jnp.log(.25 * jnp.exp(2 * result.x['log_sigma_x']) + jnp.exp(result.x['y']))
        self.C = result.x['C']
        self.d = result.x['d']
        self.W = result.x['W']
        self.update_observation_density()

    
    def _update_euclid_params2(self, X, smooth_dict):
        def objective(params, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            self.observation_density.M = jnp.array([params['C']])
            self.observation_density.b = jnp.array([params['d']])
            #self.observation_density.W = params['W']
            #self.observation_density.sigma_x = jnp.exp(params['log_sigma_x'])
            #self.observation_density.beta = .25 * jnp.exp(2 * params['log_sigma_x']) + params['y'] ** 2
            return -self.compute_Q_function(smoothing_density, X)

        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        params = {
                'C': self.C,
                'd': self.d,}
        result  = minimize(batch_objective, params, 'L-BFGS-B', args=(X, smooth_dict))
        self.C = result.x['C']
        self.d = result.x['d']
        self.update_observation_density()
        
'''   
class FullHCCovObservationModel(HCCovObservationModel):
    
    def __init__(self, Dx: int, Dz: int, Du: int, noise_x: float = 1.0):
        r"""This class implements a linear observation model, where the observations are generated as

            x_t = C z_t + d + xi_t     with      xi_t ~ N(0,Qx(z_t)),

        where

            Qx(z) = sigma_x^2 I + \sum_i U_i D_i(z) U_i',

        with D_i(z) = 2 * beta_i * cosh(h_i(z)) and h_i(z) = w_i'z + b_i.

        :param Dx: Dimensionality of observations.
        :type Dx: int
        :param Dz: Dimensionality of latent space.
        :type Dz: int
        :param Du: Number of noise directions.
        :type Du: int
        :param noise_x: Intial isoptropic std. on the observations., defaults to 1.0
        :type noise_x: float, optional
        """
        self.update_noise = 0
        assert Dx >= Du
        self.Dx, self.Dz, self.Du = Dx, Dz, Du
        if Dx == Dz:
            # self.C = jnp.eye(Dx)[:,:Dz]
            self.C = jnp.eye(Dx)
        else:
            self.C = jnp.array(np.random.randn(Dx, Dz))
            self.C = self.C / jnp.sqrt(jnp.sum(self.C**2, axis=0))[None]
        self.d = jnp.zeros(Dx)
        rand_mat = np.random.rand(self.Dx, self.Dx) - 0.5
        Q, R = np.linalg.qr(rand_mat)
        #self.U = jnp.array(Q[:, : self.Du])
        self.U = jnp.eye(self.Dx)[:,:self.Du]
        W = 1e-2 * np.random.randn(self.Du, self.Dz + 1)
        W[:, 0] = 0
        self.W = jnp.array(W)
        self.log_beta = jnp.log(jnp.ones(self.Du))#jnp.log(2 * noise_x**2 * jnp.ones(self.Du))
        self.Qx = .1**2 * jnp.eye(self.Dx)
        self.observation_density = approximate_conditional.FullHCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            Sigma=jnp.array([self.Qx]),
            U=self.U,
            W=self.W,
            beta=self.beta,
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )

    def lin_om_init(self, lin_om: LinearObservationModel):
        # self.sigma_x = jnp.array([jnp.sqrt(jnp.amin(lin_om.Qx.diagonal()))])
        beta, U = np.linalg.eig(lin_om.Qx)
        beta, U = jnp.array(beta), jnp.array(U)
        if self.Dx != 1:
            self.U = jnp.real(U[:, : self.Du])
            # self.beta = jnp.abs(jnp.real(beta[:self.Du]))
            # self.beta.at[(self.beta / self.sigma_x ** 2) < .5].set(.5 / self.sigma_x ** 2)
        #self.C = lin_om.C
        #self.d = lin_om.d
        self.Qx = lin_om.Qx
        self.update_observation_density()
        
    def update_observation_density(self):
        """Updates the emission density."""
        self.observation_density = approximate_conditional.FullHCCovGaussianConditional(
            M=jnp.array([self.C]),
            b=jnp.array([self.d]),
            Sigma=jnp.array([self.Qx]),
            U=self.U,
            W=self.W,
            beta=self.beta,
        )
        self.Qx_inv, self.ln_det_Qx = (
            self.observation_density.Lambda[0],
            self.observation_density.ln_det_Sigma[0],
        )
        
    def update_hyperparameters(
        self, X: jnp.ndarray, smooth_dict: dict, **kwargs
    ):
        """Update hyperparameters.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        
        if self.update_noise % 3 == 0:
            self._update_Qx(X, smooth_dict)
            print(self.Qx)
        elif self.update_noise % 3 == 1:
            self._update_euclid_params(X, smooth_dict)
            print(self.W)
            print(self.beta)
        elif self.update_noise % 3 == 2:
            self._update_euclid_params2(X, smooth_dict)
        if self.Dx > 1:
            self._update_stiefel_params(X, smooth_dict)
        self.update_noise += 1

        
    def _update_Qx(self, X, smooth_dict):
        Qx_batch = jit(vmap(self.get_Sigma_stats, in_axes=[0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}]))(X, smooth_dict)
        self.Qx = jnp.mean(Qx_batch, axis=0)
        print(self.Qx)
        self.update_observation_density()
        
    def get_Sigma_stats(self, X, smooth_dict):
        T = X.shape[0]
        smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
        Qx_batch = self.observation_density.get_Sigma_stats(smoothing_density, X) / T
        return Qx_batch
            
    def _update_euclid_params(self, X, smooth_dict):
        def objective(params, X, smooth_dict):
            T = X.shape[0]
            smoothing_density = pdf.GaussianPDF(**smooth_dict).slice(jnp.arange(1, T + 1))
            #self.observation_density.M = jnp.array([params['C']])
            #self.observation_density.b = jnp.array([params['d']])
            self.observation_density.W = params['W']
            self.observation_density.beta = jnp.exp(params['y']) + .25
            return -self.compute_Q_function(smoothing_density, X)

        batch_objective = lambda params, X, smooth_dict: jnp.mean(vmap(objective, in_axes=[None, 0, {'Sigma': 0, 'mu': 0, 'Lambda': 0, 'ln_det_Sigma': 0}])(params, X, smooth_dict))
        params = {'y': jnp.log(self.beta - 0.25 + 1e-10),
                  'W': self.W}
        result  = minimize(batch_objective, params, 'L-BFGS-B', args=(X, smooth_dict))
        self.log_beta = jnp.log(.25 + jnp.exp(result.x['y']) + 1e-10)
        self.W = result.x['W']
        self.update_observation_density()
'''