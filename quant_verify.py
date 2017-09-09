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
import tensorflow as tf

import data_gen
import quantifiers


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
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def run_trial(eparams, hparams, trial_num, write_dir='/tmp/tensorflow/quantexp',
        stop_loss=0.01):

    tf.reset_default_graph()

    with tf.Session() as sess, tf.variable_scope('trial_' + str(trial_num)) as scope:

        # BUILD GRAPH

        # how big each input will be
        num_quants = len(eparams['quantifiers'])
        item_size = quantifiers.Quantifier.num_chars + num_quants

        # -- input_models: [batch_size, max_len, item_size]
        input_models = tf.placeholder(tf.float32,
                [None, hparams['max_len'], item_size])
        # -- input_labels: [batch_size, num_classes]
        input_labels = tf.placeholder(tf.float32,
                [None, hparams['num_classes']])
        # -- lengths: [batch_size]
        lengths = length(input_models)

        cells = []
        for _ in range(hparams['num_layers']):
            # TODO: consider other RNN cells?
            cell = tf.contrib.rnn.LSTMCell(hparams['hidden_size'])
            # TODO: wrap in dropout?
            cells.append(cell)
        multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

        # run on input
        # -- output: [batch_size, max_len, out_size]
        output, state = tf.nn.dynamic_rnn(multi_cell,
                input_models, dtype=tf.float32, sequence_length=lengths)

        # TODO: modify to allow prediction at every time step

        # extract output at end of reading sequence
        # -- flat_output: [batch_size * max_len, out_size]
        flat_output = tf.reshape(output, [-1, hparams['hidden_size']])
        # -- indices: [batch_size]
        output_length = tf.shape(output)[0]
        indices = (tf.range(0, output_length) * hparams['max_len']
                + (lengths - 1))
        # -- final_output: [batch_size, out_size]
        final_output = tf.gather(flat_output, indices)
        tf.summary.histogram('final output', final_output)

        # make prediction
        # TODO: play with arguments here
        # -- logits: [batch_size, num_classes]
        logits = tf.contrib.layers.fully_connected(
                inputs=final_output,
                num_outputs=hparams['num_classes'])
        # -- probs: [batch_size, num_classes]
        probs = tf.nn.softmax(logits)
        # -- prediction: [batch_size]
        prediction = tf.argmax(probs, 1)
        # -- target: [batch_size]
        target = tf.argmax(input_labels, 1)

        # total accuracy
        # -- correct_prediction: [batch_size]
        correct_prediction = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
        tf.summary.scalar('total accuracy', accuracy)

        # accuracies by quantifier
        # -- flat_inputs: [batch_size * max_len, item_size]
        flat_input = tf.reshape(input_models, [-1, item_size])
        # -- final_inputs: [batch_size, item_size]
        final_inputs = tf.gather(flat_input, indices)
        # extract the portion of the input corresponding to the quantifier
        # -- quants_by_seq: [batch_size, num_quants]
        quants_by_seq = tf.slice(final_inputs,
                [0, quantifiers.Quantifier.num_chars], [-1, -1])
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

        quant_accs = []
        quant_label_dists = []
        for idx in range(num_quants):
            # -- quant_accs[idx]: accuracy for each quantifier
            quant_accs.append(
                    tf.reduce_mean(tf.to_float(
                        tf.equal(
                            prediction_by_quant[idx], target_by_quant[idx]))))
            tf.summary.scalar(
                    '{} accuracy'.format(eparams['quantifiers'][idx]._name),
                    quant_accs[idx])
            _, _, label_counts = tf.unique_with_counts(target_by_quant[idx])
            quant_label_dists.append(label_counts)

        # -- loss: [batch_size]
        loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=input_labels,
                logits=logits)
        # -- total_loss: scalar
        total_loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', total_loss)

        # training op
        # TODO: try different optimizers, parameters for it, etc
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(total_loss)

        # write summary data
        summaries = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(write_dir, sess.graph)

        # GENERATE DATA
        generator = data_gen.DataGenerator(
                hparams['max_len'], eparams['quantifiers'],
                mode=eparams['generator_mode'],
                num_data_points=eparams['num_data'])

        test_data = generator.get_test_data()
        test_models = [datum[0] for datum in test_data]
        test_labels = [datum[1] for datum in test_data]

        # TRAIN

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        accuracies = []

        # TODO: document this and section above that generates the ops
        # measures percentage of models with the same truth value
        # for every quantifier
        label_dists = sess.run(quant_label_dists, {input_models: test_models,
            input_labels: test_labels})
        for idx in range(len(label_dists)):
            print '{}: {}'.format(eparams['quantifiers'][idx]._name,
                    float(max(label_dists[idx])) / sum(label_dists[idx]))

        batch_size = eparams['batch_size']

        for epoch_idx in range(eparams['num_epochs']):

            # get training data each epoch, randomizes order
            training_data = generator.get_training_data()
            models = [data[0] for data in training_data]
            labels = [data[1] for data in training_data]

            num_batches = len(training_data) / batch_size

            for batch_idx in range(num_batches):

                batch_models = (models[batch_idx*batch_size:
                    (batch_idx+1)*batch_size])
                batch_labels = (labels[batch_idx*batch_size:
                    (batch_idx+1)*batch_size])

                sess.run(train_step,
                        {input_models: batch_models,
                            input_labels: batch_labels})

                if batch_idx % 10 == 0:
                    summary, acc, loss = sess.run([summaries, accuracy, total_loss],
                            {input_models: test_models,
                                input_labels: test_labels})
                    test_writer.add_summary(summary,
                            batch_idx + num_batches*epoch_idx)
                    accuracies.append(acc)
                    print 'Accuracy at step {}: {}'.format(batch_idx, acc)
                    print loss

                    # END TRAINING
                    # 1) very low loss, 2) accuracy convergence
                    if loss < stop_loss:
                        return
                    if batch_idx > 500 or epoch_idx > 0:
                        recent_accs = accuracies[-500:]
                        recent_avg = sum(recent_accs) / len(recent_accs)
                        print recent_avg
                        if recent_avg > 0.99:
                            return

            epoch_loss, epoch_accuracy = sess.run(
                    [total_loss, accuracy],
                    {input_models: test_models, input_labels: test_labels})
            print 'Epoch {} done'.format(epoch_idx)
            print 'Loss: {}'.format(epoch_loss)
            print 'Accuracy: {}'.format(epoch_accuracy)


# RUN AN EXPERIMENT
def experiment_one(write_dir='/tmp/tensorflow/quantexp'):

    eparams = {'num_epochs': 2, 'batch_size': 8,
            'quantifiers': [quantifiers.at_least_n(4),
                quantifiers.at_most_n(4), quantifiers.exactly_n(4)],
            'generator_mode': 'g', 'num_data': 100000}
    hparams = {'hidden_size': 24, 'num_layers': 1, 'max_len': 20,
            'num_classes': 2}
    num_trials = 20

    for idx in range(num_trials):
        run_trial(eparams, hparams, idx, write_dir)


experiment_one('data/exp1')
