import pytest
from jax import random
from timeseries_models.state_model import LSEMStateModel, LRBFMStateModel

import jax.numpy as jnp

def test_LSEMStateModel_initialization():
    model = LSEMStateModel(Dz=3, Dk=2)
    assert model.Dz == 3
    assert model.Dk == 2
    assert model.Qz.shape == (3, 3)
    assert model.A.shape == (3, 5)
    assert model.b.shape == (3,)
    assert model.W.shape == (2, 4)
    
def test_LSEMStateModel_initialization():
    model = LSEMStateModel(Dz=3, Dk=2)
    assert model.Dz == 3
    assert model.Dk == 2
    assert model.Qz.shape == (3, 3)
    assert model.A.shape == (3, 5)
    assert model.b.shape == (3,)
    assert model.W.shape == (2, 4)

def test_LRBFMStateModel_initialization():
    model = LRBFMStateModel(Dz=3, Dk=2)
    assert model.Dz == 3
    assert model.Dk == 2
    assert model.Qz.shape == (3, 3)
    assert model.A.shape == (3, 5)
    assert model.b.shape == (3,)
    assert model.mu.shape == (2, 3)
    assert model.length_scale.shape == (2, 3)

