#!/usr/bin/env python

import numpy as np

USE_GPU = False
RENDER = False
RECORD = False
ENV = 'InvertedPendulum-v1'
# ENV = 'Ant-v1'
DTYPE = np.float32
RND_SEED = 4321
EPSILON = 1e-8


SAVE_FREQ = 100000000
# UPDATE_FREQ = 100 # aka timesteps per batch
UPDATE_FREQ = 20000 # aka timesteps per batch
MAX_ITERATIONS = 500
# MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH = 20000
TEST_ITERATIONS = 100

np.random.seed(RND_SEED)
