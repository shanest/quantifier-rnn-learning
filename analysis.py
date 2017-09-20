"""
Copyright (C) 2017 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import util
# TODO: plots


COLORS = ['red', 'green', 'blue']


def experiment_one_analysis(path='data/exp1', plots=True):

    EXP1_TRIALS = range(30)
    EXP1_QUANTS = ['at_least_4', 'at_most_4', 'exactly_4']

    # read the data in
    data = util.read_trials_from_csv(path, EXP1_TRIALS)
    # FILTER OUT TRIALS WHERE RNN DID NOT LEARN
    remove_bad_trials(data)
    # get convergence points per quantifier
    convergence_points = get_convergence_points(data, EXP1_QUANTS)

    if plots:
        # make plots
        make_boxplots(convergence_points, EXP1_QUANTS)
        make_barplots(convergence_points, EXP1_QUANTS)
        make_plot(data, EXP1_QUANTS, ylim=(0.5, 1))

    # test if means are equal
    print stats.f_oneway(*[convergence_points[q] for q in EXP1_QUANTS])
    # related samples t-test; equiv to take differences, then one-sample t
    print stats.ttest_rel(convergence_points['at_least_4'], convergence_points['exactly_4'])
    print stats.ttest_rel(convergence_points['at_least_4'], convergence_points['at_most_4'])
    print stats.ttest_rel(convergence_points['at_most_4'], convergence_points['exactly_4'])


def experiment_two_analysis(path='data/exp2', plots=True):

    EXP2_TRIALS = range(30)
    EXP2_QUANTS = ['at_least_3', 'first_3']

    # read the data in
    data = util.read_trials_from_csv(path, EXP2_TRIALS)
    # FILTER OUT TRIALS WHERE RNN DID NOT LEARN
    remove_bad_trials(data)
    # get convergence points per quantifier
    convergence_points = get_convergence_points(data, EXP2_QUANTS)

    if plots:
        # make plots
        make_boxplots(convergence_points, EXP2_QUANTS)
        make_barplots(convergence_points, EXP2_QUANTS)
        make_plot(data, EXP2_QUANTS, ylim=(0.5, 1))

    # test if means are equal
    print stats.ttest_rel(convergence_points['at_least_3'], convergence_points['first_3'])


def experiment_three_analysis(path='data/exp3', plots=True):

    EXP3_TRIALS = range(30)
    EXP3_QUANTS = ['not_all', 'not_only']

    # read the data in
    data = util.read_trials_from_csv(path, EXP3_TRIALS)
    # FILTER OUT TRIALS WHERE RNN DID NOT LEARN
    remove_bad_trials(data)
    # get convergence points per quantifier
    convergence_points = get_convergence_points(data, EXP3_QUANTS)

    if plots:
        # make plots
        make_boxplots(convergence_points, EXP3_QUANTS)
        make_barplots(convergence_points, EXP3_QUANTS)
        make_plot(data, EXP3_QUANTS, ylim=(0.8, 1))

    # test if means are equal
    print stats.ttest_rel(convergence_points['not_only'], convergence_points['not_all'])


def remove_bad_trials(data):
    accuracies = [data[key]['total_accuracy'].values for key in data.keys()]
    forward_accs = [forward_means(accs) for accs in accuracies]
    threshold_pos = [first_above_threshold(accs) for accs in forward_accs]
    # a trial is bad if the forward mean never hit 0.99
    bad_trials = [idx for idx, threshold in enumerate(threshold_pos)
            if threshold is None]
    print 'Number of bad trials: {}'.format(len(bad_trials))
    for trial in bad_trials:
        del data[trial]


def get_convergence_points(data, quants):
    convergence_points = {q: [] for q in quants}
    for trial in data.keys():
        for quant in quants:
            convergence_points[quant].append(
                    data[trial]['steps'][
                        convergence_point(
                            data[trial][quant + '_accuracy'].values)])
    return convergence_points


def diff(ls1, ls2):
    assert len(ls1) == len(ls2)
    return [ls1[i] - ls2[i] for i in range(len(ls1))]


def forward_means(arr):

    return [sum(arr[idx:]) / (len(arr) - idx) for idx in range(len(arr))]


def first_above_threshold(arr, threshold=0.98):

    for idx in range(len(arr)):
        if arr[idx] > threshold:
            return idx
    return None


def convergence_point(arr, threshold=0.98):
    return first_above_threshold(forward_means(arr), threshold)


def make_plot(data, quants, ylim=None):

    assert len(quants) <= len(COLORS)

    trials_by_quant = [[] for _ in range(len(quants))]
    for trial in data.keys():
        steps = data[trial]['steps'].values
        for idx in range(len(quants)):
            trials_by_quant[idx].append(smooth_data(
                    data[trial][quants[idx] + '_accuracy'].values))
            plt.plot(steps, trials_by_quant[idx][-1],
                    COLORS[idx], alpha=0.3)

    # plot median lines
    medians_by_quant = [get_median_diff_lengths(trials_by_quant[idx])
            for idx in range(len(trials_by_quant))]
    for idx in range(len(quants)):
        # TODO: make x-axis code cleaner? recorded very 10 steps
        plt.plot([i*10 for i in range(len(medians_by_quant[idx]))],
                smooth_data(medians_by_quant[idx]),
                COLORS[idx],
                label=quants[idx],
                linewidth=2)

    if ylim:
        plt.ylim(ylim)

    plt.legend(loc=4)
    plt.show()


def get_median_diff_lengths(trials):

    max_len = np.max([len(trial) for trial in trials])
    # pad trials with NaN values to length of longest trial
    trials = np.asarray(
            [np.pad(trial, (0, max_len - len(trial)),
                'constant', constant_values=np.nan)
                for trial in trials])
    return np.nanmedian(trials, axis=0)


def make_boxplots(convergence_points, quants):

    plt.boxplot([convergence_points[quant] for quant in quants])
    plt.xticks(range(1, len(quants)+1), quants)
    plt.show()


def make_barplots(convergence_points, quants):

    means = {quant: np.mean(convergence_points[quant]) for quant in quants}
    stds = {quant: np.std(convergence_points[quant]) for quant in quants}
    intervals = {quant: stats.norm.interval(0.95,
        loc=means[quant],
        scale=stds[quant]/np.sqrt(len(convergence_points[quant])))
        for quant in quants}

    # plotting info
    index = np.arange(len(quants))
    bar_width = 0.75
    #reshape intervals to be fed to pyplot
    yerrs = [[means[quant] - intervals[quant][0] for quant in quants],
            [intervals[quant][1] - means[quant] for quant in quants]]

    plt.bar(index, [means[quant] for quant in quants], bar_width, yerr=yerrs,
            color=[COLORS[idx] for idx in range(len(quants))],
            ecolor='black', align='center')
    plt.xticks(index, quants)
    plt.ylim(ymin=0)
    plt.show()


def smooth_data(data, smooth_weight=0.9):
    prev = data[0]
    smoothed = []
    for pt in data:
        smoothed.append(prev*smooth_weight + pt*(1-smooth_weight))
        prev = smoothed[-1]
    return smoothed


experiment_two_analysis()
