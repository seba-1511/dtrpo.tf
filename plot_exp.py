#!/usr/bin/env python

import sys
import numpy as np
from math import ceil
from seb.plot import Plot, Container
from randopt import Experiment

if __name__ == '__main__':
    exp_pre = sys.argv[1]
    exp_names = sys.argv[2:]
    exp_plots = []
    exp_rewards = []
    exp_scores = []

    mean_test = Plot('Mean Test')
    mean_train = Plot('Mean Training')

    for exp_name in exp_names:
        p = Plot(exp_name)
        e = Experiment(name=exp_pre + '_' + exp_name, params={})
        e_rewards = []
        e_scores = []
        for i, res in enumerate(e.all_results()):
            score = res.params['test_reward']
            train_rewards = res.params['avg_reward']
            e_rewards.append(np.array(train_rewards))
            e_scores.append(score / float(res.params['test_n_iter']))
            p.plot(range(len(train_rewards)),
                   train_rewards, label='Run ' + str(i), alpha=0.3)
        mean_score = 0
        if len(e_scores) > 0:
            mean_score = np.mean(e_scores)
            mean_rewards = np.mean(e_rewards, axis=0)
            p.plot(range(len(mean_rewards)), mean_rewards, label='Mean')
            exp_rewards.append(mean_rewards / np.abs(np.max(mean_rewards)))
        exp_plots.append(p)
        exp_scores.append(mean_score)

    rewards_mean = np.mean(exp_rewards, axis=0)
    rewards_std = np.std(exp_rewards)
    mean_train.plot(range(len(rewards_mean)), rewards_mean, jitter=rewards_std)
    mean_test.bar(exp_scores, labels=[''.join(c for c in name if c.isupper())
                                      for name in exp_names])

    # Plot Altogether
    cont = Container(cols=2, rows=1 + int(ceil(len(exp_names) / 2.0)))
    cont.set_plot(0, 0, mean_train)
    cont.set_plot(0, 1, mean_test)
    for i, plot in enumerate(exp_plots):
        cont.set_plot(1 + (i // 2), i % 2, plot)
    cont.save('./plots/results_' + exp_pre + '.png')
