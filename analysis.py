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

import itertools as it
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import util


COLORS = ['blue', 'red']


def experiment_analysis(path, quants, trials=range(30), plots=True,
        threshold=0.95):
    """Prints statistical tests and makes plots for experiment one.

    Args:
        path: where the trials in CSV are
        plots: whether to make plots or not
    """

    # read the data in
    data = util.read_trials_from_csv(path, trials)
    # FILTER OUT TRIALS WHERE RNN DID NOT LEARN
    remove_bad_trials(data, quants, threshold=threshold)
    # get convergence points per quantifier
    convergence_points = get_convergence_points(data, quants, threshold)

    if plots:
        # make plots
        make_boxplots(convergence_points, quants)
        make_barplots(convergence_points, quants)
        make_plot(data, quants, ylim=(0.8, 1), threshold=threshold)

    print stats.ttest_rel(convergence_points[quants[0]],
                          convergence_points[quants[1]])

    final_n = 50
    final_means = [[forward_means(data[trial][quant + '_accuracy'].values,
        window_size=final_n)[-final_n] for quant in quants]
        for trial in data]
    print 'final means: {} - {}'.format(quants[0], quants[1])
    print stats.ttest_rel(
            [means[0] for means in final_means],
            [means[1] for means in final_means])


def experiment_one_a_analysis():
    experiment_analysis('data/exp1a', ['at_least_4',
                                       'at_least_6_or_at_most_2'])


def experiment_one_b_analysis():
    experiment_analysis('data/exp1b', ['at_most_3',
                                       'at_least_6_or_at_most_2'])


def experiment_two_a_analysis():
    experiment_analysis('data/exp2a', ['at_least_3', 'first_3'])


def experiment_two_b_analysis():
    experiment_analysis('data/exp2b', ['at_least_3', 'last_3'], threshold=0.93)


def experiment_three_analysis():
    experiment_analysis('data/exp3', ['not_all', 'not_only'])


def remove_bad_trials(data, quants, threshold=0.97):
    """Remove 'bad' trials from a data set.  A trial is bad if it's not
    the case that each quantifier's accuracy converged to a threshold.
    The bad trials are deleted from data, but nothing is returned.
    """
    bad_trials = set([])
    for quant in quants:
        accuracies = [data[key][quant + '_accuracy'].values for key in data]
        forward_accs = [forward_means(accs) for accs in accuracies]
        threshold_pos = [first_above_threshold(accs, threshold)
                         for accs in forward_accs]
        # a trial is bad if the forward mean never hit 0.99
        bad_trials |= set([idx for idx, thresh in enumerate(threshold_pos)
                      if thresh is None])
    print 'Number of bad trials: {}'.format(len(bad_trials))
    for trial in bad_trials:
        del data[trial]


def get_convergence_points(data, quants, threshold):
    """Get convergence points by quantifier for the data.

    Args:
        data: a dictionary, intended to be made by util.read_trials_from_csv
        quants: list of quantifier names

    Returns:
        a dictionary, with keys the quantifier names, and values the list of
        the step at which accuracy on that quantifier converged on each trial.
    """
    convergence_points = {q: [] for q in quants}
    for trial in data.keys():
        for quant in quants:
            convergence_points[quant].append(
                data[trial]['global_step'][
                    convergence_point(
                        data[trial][quant + '_accuracy'].values,
                        threshold)])
    return convergence_points


def diff(ls1, ls2):
    """List difference function.

    Args:
        ls1: first list
        ls2: second list

    Returns:
        pointwise difference ls1 - ls2
    """
    assert len(ls1) == len(ls2)
    return [ls1[i] - ls2[i] for i in range(len(ls1))]


def forward_means(arr, window_size=250):
    """Get the forward means of a list. The forward mean at index i is
    the sum of all the elements from i until i+window_size, divided
    by the number of such elements. If there are not window_size elements
    after index i, the forward mean is the mean of all elements from i
    until the end of the list.

    Args:
        arr: the list to get means of
        window_size: the size of the forward window for the mean

    Returns:
        a list, of same length as arr, with the forward means
    """
    return [(sum(arr[idx:min(idx+window_size, len(arr))])
             / min(window_size, len(arr)-idx))
            for idx in range(len(arr))]


def first_above_threshold(arr, threshold):
    """Return the point at which a list value is above a threshold.

    Args:
        arr: the list
        threshold: the threshold

    Returns:
        the first i such that arr[i] > threshold, or None if there is not one
    """
    means = forward_means(arr)
    for idx in range(len(arr)):
        if arr[idx] > threshold and means[idx] > threshold:
            return idx
    return None


def convergence_point(arr, threshold=0.95):
    """Get the point at which a list converges above a threshold.

    Args:
        arr: the list
        threshold: the threshold

    Returns:
        the first i such that forward_means(arr)[i] is above threshold
    """
    return first_above_threshold(arr, threshold)


def get_max_steps(data):
    """Gets the longest `global_step` column from a data set.

    Args:
        data: a dictionary, whose values are pandas.DataFrame, which have a
        column named `global_step`

    Returns:
        the values for the longest `global_step` column in data
    """
    max_val = None
    max_len = 0
    for key in data.keys():
        new_len = len(data[key]['global_step'].values)
        if new_len > max_len:
            max_len = new_len
            max_val = data[key]['global_step'].values
    return max_val


def make_plot(data, quants, ylim=None, threshold=0.95):
    """Makes a line plot of the accuracy of trials by quantifier, color coded,
    and with the medians also plotted.

    Args:
        data: the data
        quants: list of quantifier names
        ylim: y-axis boundaries
    """
    assert len(quants) <= len(COLORS)

    trials_by_quant = [[] for _ in range(len(quants))]
    for trial in data.keys():
        steps = data[trial]['global_step'].values
        for idx in range(len(quants)):
            trials_by_quant[idx].append(smooth_data(
                data[trial][quants[idx] + '_accuracy'].values))
            plt.plot(steps, trials_by_quant[idx][-1],
                     COLORS[idx], alpha=0.3)

    # plot median lines
    medians_by_quant = [get_median_diff_lengths(trials_by_quant[idx])
                        for idx in range(len(trials_by_quant))]
    # get x-axis of longest trial
    longest_x = get_max_steps(data)
    for idx in range(len(quants)):
        plt.plot(longest_x,
                 smooth_data(medians_by_quant[idx]),
                 COLORS[idx],
                 label=quants[idx],
                 linewidth=2)

    max_x = max([len(ls) for ls in medians_by_quant])
    plt.plot(longest_x, [threshold for _ in range(max_x)],
             linestyle='dashed', color='green')

    if ylim:
        plt.ylim(ylim)

    plt.legend(loc=4)
    plt.show()


def get_median_diff_lengths(trials):
    """Get the point-wise median of a list of lists of possibly
    different lengths.

    Args:
        trials: a list of lists, corresponding to trials

    Returns:
        a list, of the same length as the longest list in trials,
        where the list at index i contains the median of all of the
        lists in trials that are at least i long
    """
    max_len = np.max([len(trial) for trial in trials])
    # pad trials with NaN values to length of longest trial
    trials = np.asarray(
        [np.pad(trial, (0, max_len - len(trial)),
                'constant', constant_values=np.nan)
         for trial in trials])
    return np.nanmedian(trials, axis=0)


def make_boxplots(convergence_points, quants):
    """Makes box plots of some data.

    Args:
        convergence_points: dictionary of quantifier convergence points
        quants: names of quantifiers
    """
    plt.boxplot([convergence_points[quant] for quant in quants])
    plt.xticks(range(1, len(quants)+1), quants)
    plt.show()


def make_barplots(convergence_points, quants):
    """Makes bar plots, with confidence intervals, of some data.

    Args:
        convergence_points: dictionary of quantifier convergence points
        quants: names of quantifiers
    """
    pairs = list(it.combinations(quants, 2))
    assert len(pairs) <= len(COLORS)

    diffs = {pair: diff(convergence_points[pair[0]],
                        convergence_points[pair[1]])
             for pair in pairs}
    means = {pair: np.mean(diffs[pair]) for pair in pairs}
    stds = {pair: np.std(diffs[pair]) for pair in pairs}
    intervals = {pair: stats.norm.interval(
        0.95, loc=means[pair],
        scale=stds[pair]/np.sqrt(len(diffs[pair])))
        for pair in pairs}

    # plotting info
    index = np.arange(len(pairs))
    bar_width = 0.75
    # reshape intervals to be fed to pyplot
    yerrs = [[means[pair] - intervals[pair][0] for pair in pairs],
             [intervals[pair][1] - means[pair] for pair in pairs]]

    plt.bar(index, [means[pair] for pair in pairs], bar_width, yerr=yerrs,
            color=[COLORS[idx] for idx in range(len(pairs))],
            ecolor='black', align='center')
    plt.xticks(index, pairs)
    plt.show()


def smooth_data(data, smooth_weight=0.9):
    """Smooths out a series of data which might otherwise be choppy.

    Args:
        data: a line to smooth out
        smooth_weight: between 0 and 1, for 0 being no change and
            1 a flat line.  Higher values are smoother curves.

    Returns:
        a list of the same length as data, containing the smooth version.
    """
    prev = data[0]
    smoothed = []
    for point in data:
        smoothed.append(prev*smooth_weight + point*(1-smooth_weight))
        prev = smoothed[-1]
    return smoothed
