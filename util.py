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

from tensorflow.tensorboard.backend.event_processing \
        import event_accumulator as ea
import pandas as pd
# TODO: document this!


def convert_trials_to_csv(in_path, trials, out_path):
    """Takes data that was output as TF events, converts into more digestible
    CSV format and writes those files.

    Args:
        in_path: path to folder containing TF event files
        trials: names of trials to read
        out_path: path to write CSV files to
    """
    acc = ea.EventAccumulator(in_path)
    acc.Reload()

    for trial in trials:
        trial_table = get_table(acc, trial)
        trial_table.to_csv('{}/trial_{}.csv'.format(out_path, trial))


def get_table(acc, trial):
    """Generates a Pandas.DataFrame for each trial from a TF events accumulator.
    The columns are whichever scalars are found for the trial, and the steps.
    It is assumed that scalars in the event accumulator are named
    'trial_*/scalar', where '*' is the trial argument to this method.

    Args:
        acc: an EventAccumulaor
        trial: a trial id

    Returns:
        a Pandas DataFrame with all of the scalar information from trial in acc
    """
    data = {}
    scalar_tags = [tag for tag in acc.Tags()['scalars']
            if tag.startswith('trial_{}/'.format(trial))]
    # get time steps
    data['steps'] = [s.step for s in acc.Scalars(scalar_tags[0])]
    for tag in scalar_tags:
        key = tag.split('/')[1]
        values = [s.value for s in acc.Scalars(tag)]
        data[key] = values

    return pd.DataFrame(data)


def read_trials_from_csv(path, trials):
    """Reads trial information from CSV files.
    It's assumed that the files are named path/trial_X.csv, for each
    X in the list trials.

    Args:
        path: path to CSV files with trial info
        trials: list of trial IDs

    Returns:
        a dictionary, with keys being each X in trial, and values being
        Pandas DataFrames, as generated by get_table
    """
    data = {}
    for trial in trials:
        data[trial] = pd.DataFrame.from_csv(
                '{}/trial_{}.csv'.format(path, trial))
    return data
