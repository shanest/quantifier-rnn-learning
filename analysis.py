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

import scipy.stats as stats
import util
# TODO: plots


def experiment_one_analysis(path='data/exp1'):

    EXP1_TRIALS = range(20)
    EXP1_QUANTS = ['at_least_4', 'at_most_4', 'exactly_4']

    # read the data in
    data = util.read_trials_from_csv(path, EXP1_TRIALS)

    # FILTER OUT TRIALS WHERE RNN DID NOT LEARN
    accuracies = [data[key]['total_accuracy'].values for key in data.keys()]
    forward_accs = [forward_means(accs) for accs in accuracies]
    threshold_pos = [first_above_threshold(accs) for accs in forward_accs]
    # a trial is bad if the forward mean never hit 0.99
    bad_trials = [idx for idx, threshold in enumerate(threshold_pos)
            if threshold is None]
    for trial in bad_trials:
        del data[trial]

    # get convergence points per quantifier
    convergence_points = {q: [] for q in EXP1_QUANTS}
    for trial in data.keys():
        for quant in EXP1_QUANTS:
            convergence_points[quant].append(
                    data[trial]['steps'][
                        convergence_point(
                            data[trial][quant + '_accuracy'].values)])
    print convergence_points
    # test if means are equal
    print stats.f_oneway(*[convergence_points[q] for q in EXP1_QUANTS])
    # TODO: related? equiv to take differences, then one-sample t
    print stats.ttest_rel(convergence_points['at_least_4'], convergence_points['exactly_4'])
    print stats.ttest_rel(convergence_points['at_least_4'], convergence_points['at_most_4'])
    print stats.ttest_rel(convergence_points['at_most_4'], convergence_points['exactly_4'])


def diff(ls1, ls2):
    assert len(ls1) == len(ls2)
    return [ls1[i] - ls2[i] for i in range(len(ls1))]


def forward_means(arr):

    return [sum(arr[idx:]) / (len(arr) - idx) for idx in range(len(arr))]


def first_above_threshold(arr, threshold=0.99):

    for idx in range(len(arr)):
        if arr[idx] > threshold:
            return idx
    return None


def convergence_point(arr, threshold=0.99):
    return first_above_threshold(forward_means(arr), threshold)


experiment_one_analysis()
