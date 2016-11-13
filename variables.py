#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

USE_GPU = False
RENDER = False
RECORD = True
FILTER = True
FILTER_REWARDS = False
ENV = 'InvertedDoublePendulum-v1'
# ENV = 'InvertedPendulum-v1'
DTYPE = np.float32
RND_SEED = 1234 * rank
EPSILON = 1e-8
CG_DAMPING = 0.1
LAM = 0.97
GAMMA = 0.99
MAX_KL = 0.01
GAE = True


SAVE_FREQ = 10000
UPDATE_FREQ = 15000 // size # aka timesteps per batch
MAX_ITERATIONS = 50
# MAX_PATH_LENGTH = 20000
MAX_PATH_LENGTH = 5000
TEST_ITERATIONS = 100

np.random.seed(RND_SEED)
