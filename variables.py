#!/usr/bin/env python

import numpy as np

USE_GPU = False
RENDER = False
RECORD = True
# ENV = 'InvertedDoublePendulum-v1'
ENV = 'Reacher-v1'
DTYPE = np.float32
RND_SEED = 1234
EPSILON = 1e-8


SAVE_FREQ = 1000
# UPDATE_FREQ = 100 # aka timesteps per batch
UPDATE_FREQ = 20000 # aka timesteps per batch
MAX_ITERATIONS = 300
# MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH = 20000
TEST_ITERATIONS = 100

np.random.seed(RND_SEED)
