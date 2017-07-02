import numpy as np
import gc

class Quantifier(object):

    # 4 chars: A cap B, A - B, B - A, M - (A cup B)
    num_chars = 4
    #chars = one-hot encoding
    chars = np.identity(num_chars)
    #zero char for padding
    zero_char = np.zeros(num_chars)

    #name the characters, for readability
    AB = chars[0]
    AnotB = chars[1]
    BnotA = chars[2]
    neither = chars[3]

    T = (1,0)
    F = (0,1)

    #TODO: other properties of quantifiers?

    def __init__(self, name, isom=True, cons=True, lcons=False, rmon=True, lmon=None, fn=None):

        if fn is None:
            raise ValueError("supply a function for verifying a quantifier!")

        self._name = name
        self._isom = isom
        self._cons = cons
        self._lcons = lcons
        self._rmon = rmon
        self._lmon = lmon
        self._verify = fn

    def __call__(self, seq):
        return self._verify(seq)

def at_least_n_ver(seq, n):
    """Verifies whether |A cap B| > n.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff at least n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            if num_AB == n-1:
                return Quantifier.T
            else:
                num_AB += 1
    return Quantifier.F

def at_least_n(n):
    """Generates a Quantifier corresponding to at least n.

    Args: 
        n: integer

    Returns:
        Quantifier, with at_least_n_ver(_, n) as its verifier
    """
    return Quantifier("at least {}".format(n),
            isom=True, cons=True, lcons=True, rmon=True, lmon=True, 
            fn=lambda seq: at_least_n_ver(seq, n))

some = at_least_n(1)
at_least_three = at_least_n(3)

def most_ver(seq):
    """Verifies whether |A cap B| > |A - B|

    Args:
        seq: a sequence of elements from R^4

    Returns:
        Quantifier.T iff more elements are Quantifier.AB than are
        Quantifier.AnotB
    """
    diff = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            diff += 1
        elif np.array_equal(item, Quantifier.AnotB):
            diff -= 1
    return Quantifier.T if diff > 0 else Quantifier.F
most = Quantifier("most",
        isom=True, cons=True, lcons=False, rmon=True, lmon=None,
        fn=most_ver)

def first_n_ver(seq, n):
    """Verifies whether the first n As are also Bs.

    Args:
        seq: sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff the first three elements of seq that are either
        Quantifier.AB or Quantifier.AnotB are in fact Quantifier.AB.
        Quantifier.F if either seq has length less than n or there are 
        fewer than n Quantifier.ABs in seq.
    """

    # TODO: more complicated presupposition handling instead of just false?
    if len(seq) < n:
        return Quantifier.F

    num_AB = 0
    for item in seq:
        if num_AB >= n:
            return Quantifier.T
        # if an A-not-B found before n ABs are, return F
        if np.array_equal(item, Quantifier.AnotB) and num_AB < n:
            return Quantifier.F
        elif np.array_equal(item, Quantifier.AB):
            num_AB += 1

    # there are less than n ABs in total
    return Quantifier.F

def first_n(n):
    """Generates a Quantifier corresponding to `the first n'.

    Args:
        n: integer

    Returns:
        a Quantifier, with first_n_ver(_, n) as its verifier
    """
    return Quantifier("first {}".format(n),
            isom=False, cons=True, lcons=False, rmon=True, lmon=None,
            fn=lambda seq: first_n_ver(seq, n))

first_three = first_n(3)

def get_all_quantifiers():
    """Returns: a list of all Quantifiers that have been created so far.
    """
    return [quant for quant in gc.get_objects() if isinstance(quant, Quantifier)]
