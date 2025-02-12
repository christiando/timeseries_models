# Timeseries Models

A library to learn probabilistic time-series models. Mostly, the models are assumed to be conditional Gaussians, where 
means and covariances can potentially be non-linearly depending on latent variables. Models are learned with 
(approximate) expectation maximization algorithms.

# Installation 

Clone the repo and install

```bash
git clone https://github.com/christiando/timeseries_models
cd timeseries_models
pip install .
```

If you want to run the notebooks you also have to install the packages in the `requirements.txt`.

```bash
pip install -r requirements.txt
```

It is recommended to do this in an isolated python environment. Alternatively one can build the Dockerimage that comes with the repo in `.devcontainer`.


# Basic library structure

The approach of the library is to compar
State space models are constructed of a [state model](https://github.com/christiando/timeseries_models/blob/main/timeseries_models/state_model.py) 
and an [observation model](https://github.com/christiando/timeseries_models/blob/main/timeseries_models/observation_model.py). 
These can then be combined in a [state-space model](https://github.com/christiando/timeseries_models/blob/main/timeseries_models/state_space_model.py).

The state-space model can then be learnt via an (approximate) expectation maximization algorithm. 

# Example code

First we construct the state space model.
```python

# imports
from jax import config
config.update("jax_enable_x64", True) # It is recommended to run the code always with double precision.

from timeseries_models import state_model, observation_model, state_space_model

# load data, jax numpy ndarray, either [T, C] or [B, T, C]
data_train = ...

# construct model
latent_dims = 2            # Dimensions of state space
data_dims = data_train.shape[-1] # Dimensions of data space

sm = state_model.LinearStateModel(latent_dims)
om = observation_model.LinearObservationModel(data_dims, latent_dims)
ssm = state_space_model.StateSpaceModel(observation_model=om, state_model=sm)
```

Then learning the model is as simple as
```python
# Fit model
ssm.fit(data_train)
```

To do predictions with the learnt model just need to load the data and predict the model.

```python
# Predict
T_hist = 0
data_pred = ... # Load prediction data, [T_hist + T_pred, C] or [B, T_hist + T_pred, C]
result_pred = ssm.predict(data_pred, first_prediction_index = T_hist)
```
Note that this code relies heavily on the [gaussian_toolbox](https://github.com/christiando/gaussian-toolbox)

## Getting started

First checkout the [Tutorial notebook](https://github.com/christiando/timeseries_models/blob/main/notebooks/tutorial.ipynb) which shows small examples.

## Available models

### State models

The state models that are considered here, have the form

```math
\mathbf{z}_t = f_t(\mathbf{z}_{t-1}) + \zeta_t,
```
where $`\zeta_t \sim N(0,\Sigma_z(t))`$, i.e. the transition probability is normal

```math
p(\mathbf{z}_t\vert \mathbf{z}_{t-1}) = N(f_t(\mathbf{z}_{t-1}),\Sigma_z).
```

#### `LinearStateModel`

This is a linear state transition model   
```math
\mathbf{z}_t = A \mathbf{z}_{t-1} + \mathbf{b} + \zeta_t,
```
with $`\zeta_t \sim N(0,\Sigma_z)`$. The parameters that need to be inferred are $`A, b, \Sigma_z`$.

#### `LSEMStateModel`

This implements a linear+squared exponential mean (LSEM) state model     
```math
\mathbf{z}_t = A f(\mathbf{z}_{t-1}) + b + \zeta_t,
```
with $`\zeta_t \sim N(0,\Sigma_z)`$. The feature function is 
```math
f(\mathbf{z}) = (z_0, z_1,...,z_m, k(h_1(\mathbf{z}))),...,k(h_n(\mathbf{z}))).
```
The kernel and linear activation function are given by
$`k(h) = \exp(-h^2 / 2)`$ and $`h_i(x) = w_i'x + w_{i,0}`$.

The parameters that need to be inferred are $`A, b, \Sigma_z, W`$, where $`W`$ are all the kernel weights.

#### `LRBFMStateModel`

This implements a linear+radial basis function mean (LRBF) state model     
```math
\mathbf{z}_t = A f(\mathbf{z}_{t-1}) + b + \zeta_t,
```
with $`\zeta_t \sim N(0,\Sigma_z)`$. The feature function is 
```math
f(\mathbf{z}) = (z_0, z_1,...,z_m, k_1(\mathbf{z}),...,k_n(\mathbf{z})).
```
The kernel and linear activation function are given by
$`k_i(z) = \exp(-|z-c_i|^2_2 / 2\sigma_i^2)`$.

The parameters that need to be inferred are $`A, b, \Sigma_z, c,\sigma`$.

### Observation models

The observation models that are considered here, have the form

```math
\mathbf{c}_t = g_t(\mathbf{z}_{t}) + \xi_t,
```
where $`\xi_t \sim N(0,\Sigma_x(t))`$, i.e. the transition probability is normal

```math
p(\mathbf{x}_t\vert \mathbf{z}_{t}) = N(g_t(\mathbf{z}_{t}),\Sigma_z).
```

#### `LinearObservationModel`

```math
\mathbf{x}_t = C \mathbf{z}_t + \mathbf{d} + \xi_t 
```
with $`\xi_t \sim N(0,\Sigma_x)`$. Parameters to be inferred are $`C, \mathbf{d}, \Sigma_x`$.

#### `LSEMObservationModel`

Non-linearity has the same form of $f$ in `LSEMStateModel`.

#### `LRBFMObservationModel`

Non-linearity has the same form of $f$ in `LRBFMStateModel`.

# Citation

The library was mainly developed for the [publication](https://doi.org/10.1016/j.ijforecast.2025.01.002). If you use the library please cite

```
@article{DONNER2025,
title = {A projected nonlinear state-space model for forecasting time series signals},
journal = {International Journal of Forecasting},
year = {2025},
issn = {0169-2070},
doi = {https://doi.org/10.1016/j.ijforecast.2025.01.002},
url = {https://www.sciencedirect.com/science/article/pii/S0169207025000020},
author = {Christian Donner and Anuj Mishra and Hideaki Shimazaki}
}
```