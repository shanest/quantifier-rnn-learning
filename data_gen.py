import numpy as np
import itertools
import math
import quantifiers

#TODO: move batching logic from quant_verify.run_experiment to here?
#TODO: allow reading/writing data to files instead of in memory?

class DataGenerator(object):

    def __init__(self, max_len, quants=quantifiers.get_all_quantifiers(), training_split=0.7):

        self._quantifiers = quants
        self._models = self._generate_sequences(max_len)
        self._labeled_data = self._generate_labeled_data()
        self._training_split = 0.7
        self._training_data = None
        self._test_data = None

    def _generate_sequences(self, max_len):
        """Generates all sequences of characters up to length max_len.
        These correspond to finite models. Shorter sequences are padded with
        a dummy zero character to make all sequences have the same length.

        Args:
            max_len: the maximum length of a sequence (aka size of a model)

        Returns: 
            a list of all sequences of characters up to length max_len
        """

        all_seqs = []

        for n in range(1, max_len+1):
            seqs = list(itertools.product(quantifiers.Quantifier.chars, repeat=n))
            # pad with zero character
            if n < max_len:
                seqs = [tup + (quantifiers.Quantifier.zero_char,)*(max_len - n) for tup in seqs]
            all_seqs.extend(seqs)

        return all_seqs

    def _generate_labeled_data(self):

        num_quants = len(self._quantifiers)
        quant_labels = np.identity(num_quants)
        self._labeled_data = []

        #TODO: more quantifiers here, as part of data...
        for idx in range(num_quants):
            self._labeled_data.extend(
                    [( [np.concatenate([char, quant_labels[idx]]) for char in seq] , 
                        self._quantifiers[idx](seq))
                        for seq in self._models])

        np.random.shuffle(self._labeled_data)
        return self._labeled_data

    def get_training_data(self):
        """Gets training data, based on the percentage self._training_split.
        Shuffles the training data every time it is called.
        """

        if self._training_data is None:
            idx = int(math.ceil(self._training_split * len(self._labeled_data)))
            self._training_data = self._labeled_data[:idx]
            
        np.random.shuffle(self._training_data)
        return self._training_data

    def get_test_data(self):
        """Gets test data, based on the percentage 1 - self._training_split.
        """

        if self._test_data is None:
            idx = int(math.ceil(self._training_split * len(self._labeled_data)))
            self._test_data = self._labeled_data[idx:]

        return self._test_data
