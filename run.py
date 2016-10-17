#!/usr/bin/env python

import gym
from time import time

from variables import (MAX_ITERATIONS, ENV, RENDER, SAVE_FREQ, TEST_ITERATIONS,
                       MAX_PATH_LENGTH, RND_SEED, UPDATE_FREQ, RECORD)
from trpo import TRPO
from utils import FCNet, numel
from optimizers import ConjugateGradients


if __name__ == '__main__':
    env = gym.make(ENV)
    env.seed(RND_SEED)

    policy = FCNet(numel(env.observation_space),
                   numel(env.action_space))
    opt = ConjugateGradients()
    agent = TRPO(env, policy, optimizer=opt, update_freq=UPDATE_FREQ)

    # Train Time:
    training_start = time()
    while agent.n_iterations < MAX_ITERATIONS:
        state = env.reset()
        for path in xrange(MAX_PATH_LENGTH):
            action, action_info = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if RENDER:
                env.render()
            agent.learn(state, action, reward, next_state, done, action_info)
            if agent.n_iterations % SAVE_FREQ == 1:
                agent.save('./snapshots/trpo' + str(time()) + '.pkl')
            if done or agent.done():
                break
        if agent.done():
            break

    training_end = time()

    # Test Time:
    if RECORD:
        env.monitor.start('./videos/' + ENV + str(time()) + '/')
    test_rewards = 0
    test_start = time()
    for iteration in xrange(TEST_ITERATIONS):
        state = env.reset()
        while True:
            action, _ = agent.act(state)
            state, reward, done, _ = env.step(action)
            test_rewards += reward
            if done:
                break
    test_end = time()
    if RECORD:
        env.monitor.close()
    print '\n'
    print 'Training Time: ', training_end - training_start
    print 'Testing Time: ', test_end - test_start
    print 'Average Test Reward:', test_rewards / float(TEST_ITERATIONS)

