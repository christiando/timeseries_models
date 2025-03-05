from jax import random
from jax import numpy as jnp

from timeseries_models.observation_model import LinearObservationModel, LSEMObservationModel, LRBFMObservationModel


def test_linear_observation_model_init_eye():
    model = LinearObservationModel(Dx=3, Dz=3)
    assert model.Dx == 3
    assert model.Dz == 3
    assert jnp.array_equal(model.C, jnp.eye(3))
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.delta == 0

def test_linear_observation_model_init_random():
    key = random.PRNGKey(0)
    model = LinearObservationModel(Dx=3, Dz=2, key=key)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.C.shape == (3, 2)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.delta == 0

def test_linear_observation_model_init_noise():
    model = LinearObservationModel(Dx=3, Dz=3, noise_x=2.0)
    assert model.Dx == 3
    assert model.Dz == 3
    assert jnp.array_equal(model.C, jnp.eye(3))
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, 4.0 * jnp.eye(3))
    assert model.delta == 0

def test_linear_observation_model_init_delta():
    model = LinearObservationModel(Dx=3, Dz=3, delta=1.0)
    assert model.Dx == 3
    assert model.Dz == 3
    assert jnp.array_equal(model.C, jnp.eye(3))
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, 2.0 * jnp.eye(3))
    assert model.delta == 1.0

def test_lsem_observation_model_init():
    model = LSEMObservationModel(Dx=3, Dz=2, Dk=4)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.lambda_W == 0.0

def test_lsem_observation_model_init_noise():
    model = LSEMObservationModel(Dx=3, Dz=2, Dk=4, noise_x=2.0)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, 4.0 * jnp.eye(3))
    assert model.lambda_W == 0.0

def test_lsem_observation_model_init_lambda_W():
    model = LSEMObservationModel(Dx=3, Dz=2, Dk=4, lambda_W=0.5)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.lambda_W == 0.5

def test_lsem_observation_model_init_random():
    key = random.PRNGKey(0)
    model = LSEMObservationModel(Dx=3, Dz=2, Dk=4, key=key)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.lambda_W == 0.0

def test_lrbfm_observation_model_init():
    model = LRBFMObservationModel(Dx=3, Dz=2, Dk=4)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.kernel_type == "isotropic"

def test_lrbfm_observation_model_init_noise():
    model = LRBFMObservationModel(Dx=3, Dz=2, Dk=4, noise_z=2.0)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, 4.0 * jnp.eye(3))
    assert model.kernel_type == "isotropic"

def test_lrbfm_observation_model_init_kernel_type_scalar():
    model = LRBFMObservationModel(Dx=3, Dz=2, Dk=4, kernel_type="scalar")
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.kernel_type == "scalar"

def test_lrbfm_observation_model_init_kernel_type_anisotropic():
    model = LRBFMObservationModel(Dx=3, Dz=2, Dk=4, kernel_type="anisotropic")
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.kernel_type == "anisotropic"

def test_lrbfm_observation_model_init_random():
    key = random.PRNGKey(0)
    model = LRBFMObservationModel(Dx=3, Dz=2, Dk=4, key=key)
    assert model.Dx == 3
    assert model.Dz == 2
    assert model.Dk == 4
    assert model.Dphi == 6
    assert model.C.shape == (3, 6)
    assert jnp.array_equal(model.d, jnp.zeros(3))
    assert jnp.array_equal(model.Qx, jnp.eye(3))
    assert model.kernel_type == "isotropic"


