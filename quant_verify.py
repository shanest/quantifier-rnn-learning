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
from collections import defaultdict
import argparse
import tensorflow as tf
import numpy as np

import data_gen
import quantifiers
import util


INPUT_FEATURE = 'x'


# for variable length sequences,
# see http://danijar.com/variable-sequence-lengths-in-tensorflow/
def length(data):
    """Gets real length of sequences from a padded tensor.

    Args:
        data: a Tensor, containing sequences

    Returns:
        a Tensor, of shape [data.shape[0]], containing the length
        of each sequence
    """
    data = tf.slice(data,
                    [0, 0, 0],
                    [-1, -1, quantifiers.Quantifier.num_chars])
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(length, tf.int32)
    return lengths


# TODO: some docs here, noting TF estimator stuff
def lstm_model_fn(features, labels, mode, params):

    # BUILD GRAPH

    # how big each input will be
    num_quants = len(params['quantifiers'])
    item_size = quantifiers.Quantifier.num_chars + num_quants

    # -- input_models: [batch_size, max_len, item_size]
    input_models = features[INPUT_FEATURE]
    # -- input_labels: [batch_size, num_classes]
    input_labels = labels
    # -- lengths: [batch_size], how long each input really is
    lengths = length(input_models)

    cells = []
    for _ in range(params['num_layers']):
        # TODO: consider other RNN cells?
        cell = tf.nn.rnn_cell.LSTMCell(params['hidden_size'])
        # dropout
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, state_keep_prob=params['dropout'])
        cells.append(cell)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # run on input
    # -- output: [batch_size, max_len, out_size]
    output, _ = tf.nn.dynamic_rnn(
        multi_cell, input_models,
        dtype=tf.float64, sequence_length=lengths)

    # TODO: modify to allow prediction at every time step

    # extract output at end of reading sequence
    # -- flat_output: [batch_size * max_len, out_size]
    flat_output = tf.reshape(output, [-1, params['hidden_size']])
    # -- indices: [batch_size]
    output_length = tf.shape(output)[0]
    indices = (tf.range(0, output_length) * params['max_len']
               + (lengths - 1))
    # -- final_output: [batch_size, out_size]
    final_output = tf.gather(flat_output, indices)
    tf.summary.histogram('final_output', final_output)

    # make prediction
    # TODO: play with arguments here
    # -- logits: [batch_size, num_classes]
    logits = tf.contrib.layers.fully_connected(
        inputs=final_output,
        num_outputs=params['num_classes'],
        activation_fn=None)
    # -- probs: [batch_size, num_classes]
    probs = tf.nn.softmax(logits)
    # dictionary of outputs
    outputs = {'probs': probs}

    # exit before labels are used when in predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=outputs)

    # -- loss: [batch_size]
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=input_labels,
        logits=logits)
    # -- total_loss: scalar
    total_loss = tf.reduce_mean(loss)

    # training op
    # TODO: try different optimizers, parameters for it, etc
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(total_loss,
                                  global_step=tf.train.get_global_step())

    # total accuracy
    # -- prediction: [batch_size]
    prediction = tf.argmax(probs, 1)
    # -- target: [batch_size]
    target = tf.argmax(input_labels, 1)

    # list of metrics for evaluation
    eval_metrics = {'total_accuracy': tf.metrics.accuracy(target, prediction)}

    # metrics by quantifier
    # -- flat_inputs: [batch_size * max_len, item_size]
    flat_input = tf.reshape(input_models, [-1, item_size])
    # -- final_inputs: [batch_size, item_size]
    final_inputs = tf.gather(flat_input, indices)
    # extract the portion of the input corresponding to the quantifier
    # -- quants_by_seq: [batch_size, num_quants]
    quants_by_seq = tf.slice(final_inputs,
                             [0, quantifiers.Quantifier.num_chars],
                             [-1, -1])
    # index, in the quantifier list, of the quantifier for each data point
    # -- quant_indices: [batch_size]
    quant_indices = tf.to_int32(tf.argmax(quants_by_seq, 1))
    # -- prediction_by_quant: a list num_quants long
    # -- prediction_by_quant[i]: Tensor of predictions for quantifier i
    prediction_by_quant = tf.dynamic_partition(
        prediction, quant_indices, num_quants)
    # -- target_by_quant: a list num_quants long
    # -- target_by_quant[i]: Tensor containing true for quantifier i
    target_by_quant = tf.dynamic_partition(
        target, quant_indices, num_quants)

    for idx in xrange(num_quants):
        key = '{}_accuracy'.format(params['quantifiers'][idx]._name)
        eval_metrics[key] = tf.metrics.accuracy(
            target_by_quant[idx], prediction_by_quant[idx])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics)


class EvalEarlyStopHook(tf.train.SessionRunHook):
    """Evaluates estimator during training and implements early stopping.

    Writes output of a trial as CSV file.

    See https://stackoverflow.com/questions/47137061/. """

    def __init__(self, estimator, eval_input, filename,
                 num_steps=50, stop_loss=0.02):

        self._estimator = estimator
        self._input_fn = eval_input
        self._num_steps = num_steps
        self._stop_loss = stop_loss
        # store results of evaluations
        self._results = defaultdict(list)
        self._filename = filename

    def begin(self):

        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise ValueError("global_step needed for EvalEarlyStop")

    def before_run(self, run_context):

        requests = {'global_step': self._global_step_tensor}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):

        global_step = run_values.results['global_step']
        if (global_step-1) % self._num_steps == 0:
            ev_results = self._estimator.evaluate(input_fn=self._input_fn)

            print ''
            for key, value in ev_results.items():
                self._results[key].append(value)
                print '{}: {}'.format(key, value)

            # TODO: add running total accuracy or other complex stop condition?
            if ev_results['loss'] < self._stop_loss:
                run_context.request_stop()

    def end(self, session):
        # write results to csv
        util.dict_to_csv(self._results, self._filename)


def run_trial(eparams, hparams, trial_num,
              write_path='/tmp/tensorflow/quantexp'):

    tf.reset_default_graph()

    write_dir = '{}/trial_{}'.format(write_path, trial_num)
    csv_file = '{}/trial_{}.csv'.format(write_path, trial_num)

    # BUILD MODEL
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=eparams['eval_steps'],
        save_checkpoints_secs=None,
        save_summary_steps=eparams['eval_steps'])

    model = tf.estimator.Estimator(
        model_fn=lstm_model_fn,
        params=hparams,
        model_dir=write_dir,
        config=run_config)

    # GENERATE DATA
    generator = data_gen.DataGenerator(
        hparams['max_len'], hparams['quantifiers'],
        mode=eparams['generator_mode'],
        num_data_points=eparams['num_data'])

    training_data = generator.get_training_data()
    test_data = generator.get_test_data()

    def get_np_data(data):
        x_data = np.array([datum[0] for datum in data])
        y_data = np.array([datum[1] for datum in data])
        return x_data, y_data

    # input fn for training
    train_x, train_y = get_np_data(training_data)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: train_x},
        y=train_y,
        batch_size=eparams['batch_size'],
        num_epochs=eparams['num_epochs'],
        shuffle=True)

    # input fn for evaluation
    test_x, test_y = get_np_data(test_data)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: test_x},
        y=test_y,
        batch_size=len(test_x),
        shuffle=False)

    print '\n------ TRIAL {} -----'.format(trial_num)

    # train and evaluate model together, using the Hook
    model.train(input_fn=train_input_fn,
                hooks=[EvalEarlyStopHook(model, eval_input_fn, csv_file,
                                         eparams['eval_steps'],
                                         eparams['stop_loss'])])


# DEFINE AN EXPERIMENT

def experiment_one_a(write_dir='data/exp1a'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_least_n(4),
                               quantifiers.at_least_n_or_at_most_m(6, 2)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_one_b(write_dir='data/exp1b'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_most_n(3),
                               quantifiers.at_least_n_or_at_most_m(6, 2)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_one_c(write_dir='data/exp1c'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_least_n(4),
                               quantifiers.between_m_and_n(6, 10)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_one_d(write_dir='data/exp1d'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 100000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_most_n(4),
                               quantifiers.between_m_and_n(6, 10)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_two(write_dir='data/exp2'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 200000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.first_n(3),
                               quantifiers.at_least_n(3)]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


def experiment_three(write_dir='data/exp3'):

    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 300000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.nall, quantifiers.notonly]}
    num_trials = 30

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


# TEST
def test():
    eparams = {'num_epochs': 4, 'batch_size': 8,
               'generator_mode': 'g', 'num_data': 10000,
               'eval_steps': 50, 'stop_loss': 0.02}
    hparams = {'hidden_size': 12, 'num_layers': 2, 'max_len': 20,
               'num_classes': 2, 'dropout': 1.0,
               'quantifiers': [quantifiers.at_least_n(4),
                               quantifiers.most]}
    for idx in range(2):
        run_trial(eparams, hparams, idx)


if __name__ == '__main__':

    # RUN AN EXPERIMENT, with command-line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='which experiment to run', type=str)
    parser.add_argument('--out_path', help='path to output', type=str)
    args = parser.parse_args()

    func_map = {
        'one_a': experiment_one_a,
        'one_b': experiment_one_b,
        'two': experiment_two,
        'three': experiment_three,
        'test': test
    }
    func = func_map[args.exp]

    if args.out_path:
        func(args.out_path)
    else:
        func()
