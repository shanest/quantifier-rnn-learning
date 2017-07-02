import numpy as np
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
        a Tensor, of shape [data.shape[0]], containing the length of each sequence
    """
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def run_experiment(eparams, hparams, write_dir='/tmp/tensorflow/quantexp'):

    with tf.Session() as sess:

        # BUILD GRAPH

        #one 
        num_quants = len(eparams['quantifiers'])
        item_size = quantifiers.Quantifier.num_chars + num_quants

        # -- input_models: [batch_size, max_len, num_chars] 
        input_models = tf.placeholder(tf.float32, [None, hparams['max_len'], item_size])
        # -- input_labels: [batch_size, num_classes]
        input_labels = tf.placeholder(tf.float32, [None, hparams['num_classes']])
        # -- lengths: [batch_size]
        lengths = length(input_models)

        #TODO: consider other RNN cells?
        cell = tf.contrib.rnn.LSTMCell(hparams['hidden_size'])

        # run on input
        # -- output: [batch_size, max_len, out_size]
        output, state = tf.nn.dynamic_rnn(cell, input_models, dtype=tf.float32, sequence_length=lengths)

        #TODO: modify to allow prediction at every time step, in order to examine how network works

        # extract output at end of reading sequence
        # -- flat_output: [batch_size * max_len, out_size]
        flat_output = tf.reshape(output, [-1, hparams['hidden_size']])
        # -- indices: [batch_size]
        output_length = tf.shape(output)[0]
        indices = tf.range(0, output_length) * hparams['max_len'] + (lengths - 1)
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
        quants_by_seq = tf.slice(final_inputs, [0, quantifiers.Quantifier.num_chars], [-1, -1])
        # index, in the list of quantifiers, of the quantifier for each data point
        # -- quant_indices: [batch_size]
        quant_indices = tf.to_int32(tf.argmax(quants_by_seq, 1))
        # -- prediction_by_quant: a list num_quants long
        # -- prediction_by_quant[i]: Tensor containing predictions for quantifier i
        prediction_by_quant = tf.dynamic_partition(prediction, quant_indices, num_quants)
        # -- target_by_quant: a list num_quants long
        # -- target_by_quant[i]: Tensor containing predictions for quantifier i
        target_by_quant = tf.dynamic_partition(target, quant_indices, num_quants)

        quant_accs = []
        for idx in range(num_quants):
            # -- quant_accs[idx]: accuracy for each quantifier
            quant_accs.append(
                    tf.reduce_mean(tf.to_float(
                        tf.equal(prediction_by_quant[idx], target_by_quant[idx]))))
            tf.summary.scalar('{} accuracy'.format(eparams['quantifiers'][idx]._name), quant_accs[idx])

        # -- loss: [batch_size]
        loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=input_labels,
                logits=logits)
        # -- total_loss: scalar
        total_loss = tf.reduce_sum(loss)
        tf.summary.scalar('loss', total_loss)

        # training op
        #TODO: try different optimizers, parameters for it, etc
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(total_loss)

        # write summary data
        summaries = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(write_dir, sess.graph)


        # GENERATE DATA
        generator = data_gen.DataGenerator(hparams['max_len'], eparams['quantifiers'])

        test_data = generator.get_test_data()
        test_models = [datum[0] for datum in test_data]
        test_labels = [datum[1] for datum in test_data]

        # TRAIN

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        batch_size = eparams['batch_size']

        for epoch_idx in range(eparams['num_epochs']):

            # get training data each epoch, randomizes order
            training_data = generator.get_training_data()
            models = [data[0] for data in training_data]
            labels = [data[1] for data in training_data]

            num_batches = len(training_data) / batch_size

            for batch_idx in range(num_batches):

                batch_models = models[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_labels = labels[batch_idx*batch_size:(batch_idx+1)*batch_size]

                sess.run(train_step, {input_models: batch_models, input_labels: batch_labels})

                if batch_idx % 10 == 0:
                    summary, acc = sess.run([summaries, accuracy], {input_models: test_models, input_labels: test_labels}) 
                    test_writer.add_summary(summary, batch_idx + num_batches*epoch_idx)
                    print 'Accuracy at step {}: {}'.format(batch_idx, acc)

            epoch_loss, epoch_accuracy = sess.run([total_loss, update_accuracy], {input_models: test_models, input_labels: test_labels})
            print 'Epoch {} done'.format(epoch_idx)
            print 'Loss: {}'.format(epoch_loss)
            print 'Accuracy: {}'.format(epoch_accuracy)

# RUN AN EXPERIMENT
run_experiment(
        {'num_epochs': 4, 'batch_size': 8, 'quantifiers': quantifiers.get_all_quantifiers()},
        {'hidden_size': 32, 'num_layers': 1, 'max_len': 8, 'num_classes': 2},
)
