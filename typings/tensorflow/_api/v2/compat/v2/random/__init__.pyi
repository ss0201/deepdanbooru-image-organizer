"""
This type stub file was generated by pyright.
"""

import sys as _sys
from . import experimental
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.ops.candidate_sampling_ops import all_candidate_sampler, fixed_unigram_candidate_sampler, learned_unigram_candidate_sampler, log_uniform_candidate_sampler, uniform_candidate_sampler
from tensorflow.python.ops.random_ops import categorical, random_gamma as gamma, random_normal as normal, random_poisson_v2 as poisson, random_shuffle as shuffle, random_uniform as uniform, truncated_normal
from tensorflow.python.ops.stateful_random_ops import Generator, create_rng_state, get_global_generator, set_global_generator
from tensorflow.python.ops.stateless_random_ops import Algorithm, stateless_categorical, stateless_parameterized_truncated_normal, stateless_random_binomial as stateless_binomial, stateless_random_gamma as stateless_gamma, stateless_random_normal as stateless_normal, stateless_random_poisson as stateless_poisson, stateless_random_uniform as stateless_uniform, stateless_truncated_normal

"""Public API for tf.random namespace.
"""