#!/usr/bin/env python

import numpy as np

USE_GPU = False
RENDER = False
RECORD = False
ENV = 'InvertedPendulum-v1'
DTYPE = np.float32
RND_SEED = 1234
EPSILON = 1e-8


SAVE_FREQ = 10
UPDATE_FREQ = 100 # aka timesteps per batch
MAX_ITERATIONS = 30
MAX_PATH_LENGTH = 1000
TEST_ITERATIONS = 100

np.random.seed(RND_SEED)
