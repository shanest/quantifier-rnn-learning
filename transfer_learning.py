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
from __future__ import print_function
from builtins import range
import tensorflow as tf
import numpy as np

import data_gen
import quantifiers
import quant_verify


# EXAMPLE FOR TRANSFER LEARNING
# TODO: better documentation

def transfer_lstm_model_fn(features, labels, mode, params):

    # how big each input will be
    num_quants = len(params['quantifiers'])
    item_size = quantifiers.Quantifier.num_chars + num_quants

    # -- input_models: [batch_size, max_len, item_size]
    input_models = features[quant_verify.INPUT_FEATURE]
    # -- input_labels: [batch_size, num_classes]
    input_labels = labels
    # -- lengths: [batch_size], how long each input really is
    lengths = quant_verify.length(input_models)

    # NOTE: (i) any code which defines variables that you want to load from a
    # saved model should be put inside this variable scope
    # (ii) the code here should mirror the code used in the model_fn that
    # you used for training a model [i.e. lstm_model_fn above in this case]
    with tf.variable_scope('transferred'):
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

    # see https://github.com/tensorflow/tensorflow/issues/14713#issuecomment-349477017
    # NOTE: (iii) if you use your own model_fn and want to do transfer
    # learning, you can also put the relevant code in its own variable scope in
    # the other model_fn.  If you do that, make sure to pass the old scope in
    # as params['old_scope']
    tf.train.init_from_checkpoint(params['checkpoint_path'],
                                  {params['old_scope']: 'transferred/'})

    # do stuff with output
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

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op)


def transfer_test():

    params = {'checkpoint_path':
              'data/exp1a/trial_0/',
              'old_scope': '/',
              'hidden_size': 12, 'num_layers': 2,
              'max_len': 20, 'num_classes': 2,
              'dropout': 1.0,
              'quantifiers':
              [quantifiers.at_least_n(4),
               quantifiers.at_least_n_or_at_most_m(6, 2)]}
    model = tf.estimator.Estimator(model_fn=transfer_lstm_model_fn,
                                   params=params)

    generator = data_gen.DataGenerator(params['max_len'],
                                       params['quantifiers'],
                                       num_data_points=1000)

    some_data = generator.get_training_data()

    # must train before predict, so we'll give it one example
    model.train(input_fn=tf.estimator.inputs.numpy_input_fn(
        x={quant_verify.INPUT_FEATURE: np.array([some_data[0][0]])},
        y=np.array([some_data[0][1]]),
        shuffle=False))

    some_inputs = np.array([datum[0] for datum in some_data])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={quant_verify.INPUT_FEATURE: some_inputs},
        shuffle=False)

    predictions = list(model.predict(input_fn=predict_input_fn))
    for idx in range(5):
        print('input: {}\nprobs: {}\n'.format(some_inputs[idx],
                                              predictions[idx]['probs']))


if __name__ == '__main__':
    transfer_test()
