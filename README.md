# ModelAgnosticGWB
Model Agnostic GWB

* To run things with the latest info:

1. Install with `conda env create -f env.yml` This will install all the latest dependencies, from the correct place, so that `popstock` will work with JAX (from the `meyers-academic` fork and `jax` branch), and the newest version of `westley` is installed for parallel tempering.

2. The notebook you want is in `notebooks/test_westley.py`. It should have in it the parallel tempering example, with the latest sampler which uses jax and should be faster even for a single temperature due to reduced sampler overhead. It also includes an example with the old sampler, which is included in this repo in `modules/TransDimensionalSplineFitter` as we are used to.