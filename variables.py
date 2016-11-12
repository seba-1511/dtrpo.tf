#!/usr/bin/env python

import numpy as np

USE_GPU = False
RENDER = False
RECORD = False
FILTER = True
FILTER_REWARDS = False
ENV = 'InvertedDoublePendulum-v1'
ENV = 'InvertedPendulum-v1'
DTYPE = np.float32
RND_SEED = 1234
EPSILON = 1e-8
CG_DAMPING = 0.1
LAM = 1.0
GAMMA = 0.99
MAX_KL = 0.01


SAVE_FREQ = 10000
UPDATE_FREQ = 10000 # aka timesteps per batch
MAX_ITERATIONS = 15
# MAX_PATH_LENGTH = 20000
MAX_PATH_LENGTH = 5000
TEST_ITERATIONS = 100

np.random.seed(RND_SEED)
