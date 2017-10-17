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
import gc


class Quantifier(object):

    # 4 chars: A cap B, A - B, B - A, M - (A cup B)
    num_chars = 4
    # chars = one-hot encoding
    chars = np.identity(num_chars)
    # zero char for padding
    zero_char = np.zeros(num_chars)

    # name the characters, for readability
    AB = chars[0]
    AnotB = chars[1]
    BnotA = chars[2]
    neither = chars[3]

    T = (1, 0)
    F = (0, 1)

    # TODO: other properties of quantifiers?

    def __init__(self, name, isom=True, cons=True, lcons=False,
            rmon=True, lmon=None, fn=None):

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


def all_ver(seq):
    """Verifies whether every A is a B in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there are no Quantifier.AnotBs in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.AnotB):
            return Quantifier.F
    return Quantifier.T


every = Quantifier("every",
        isom=True, cons=True, lcons=False, rmon=True, lmon=False,
        fn=all_ver)


def notall_ver(seq):
    """Verifies whether not all As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there is a Quantifier.AnotB in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.AnotB):
            return Quantifier.T
    return Quantifier.F


nall = Quantifier("not all",
        isom=True, cons=True, lcons=False, rmon=False, lmon=True,
        fn=notall_ver)


def no_ver(seq):
    """Verifies whether no As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there is not a Quantifier.AB in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            return Quantifier.F
    return Quantifier.T


no = Quantifier("no",
        isom=True, cons=True, lcons=False, rmon=False, lmon=False,
        fn=no_ver)


def only_ver(seq):
    """Verifies whether only As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there are no Quantifier.BnotAs in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.BnotA):
            return Quantifier.F
    return Quantifier.T


only = Quantifier("only",
        isom=True, cons=False, lcons=True, rmon=False, lmon=True,
        fn=only_ver)


def notonly_ver(seq):
    """Verifies whether not only As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there is a Quantifier.BnotA in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.BnotA):
            return Quantifier.T
    return Quantifier.F


notonly = Quantifier("not only",
        isom=True, cons=False, lcons=True, rmon=True, lmon=False,
        fn=notonly_ver)


def even_ver(seq):
    """Verifies whether the number of As that are B is even.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff the number of Quantifier.ABs in seq is even
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
    if num_AB % 2 == 0:
        return Quantifier.T
    else:
        return Quantifier.F


even = Quantifier("even",
        isom=True, cons=True, lcons=True, rmon=None, lmon=None,
        fn=even_ver)


def odd_ver(seq):
    """Verifies whether the number of As that are B is odd.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff the number of Quantifier.ABs in seq is odd
    """
    return Quantifier.T if even_ver(seq) == Quantifier.F else Quantifier.F


odd = Quantifier("odd",
        isom=True, cons=True, lcons=True, rmon=None, lmon=None,
        fn=odd_ver)


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


def at_most_n_ver(seq, n):
    """Verifies whether |A cap B| <= n.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff exactly n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            if num_AB == n:
                return Quantifier.F
            else:
                num_AB += 1
    return Quantifier.T


def at_most_n(n):
    """Generates a Quantifier corresponding to at most n.

    Args:
        n: integer

    Returns:
        Quantifier, with at_most_n_ver(_, n) as its verifier
    """
    return Quantifier("at most {}".format(n),
            isom=True, cons=True, lcons=True, rmon=False, lmon=False,
            fn=lambda seq: at_most_n_ver(seq, n))


def exactly_n_ver(seq, n):
    """Verifies whether |A cap B| = n.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff exactly n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
    return Quantifier.T if num_AB == n else Quantifier.F


def exactly_n(n):
    """Generates a Quantifier corresponding to at least n.

    Args:
        n: integer

    Returns:
        Quantifier, with exactly_n_ver(_, n) as its verifier
    """
    return Quantifier("exactly {}".format(n),
            isom=True, cons=True, lcons=True, rmon=None, lmon=None,
            fn=lambda seq: exactly_n_ver(seq, n))


exactly_three = exactly_n(3)


def all_but_n_ver(seq, n):
    """Verifies whether |A - B| = 4.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff exactly n elements are Quantifier.AnotB
    """
    num_AnotB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AnotB):
            num_AnotB += 1
    return Quantifier.T if num_AnotB == n else Quantifier.F


def all_but_n(n):
    """Generates a Quantifier corresponding to all but n.

    Args:
        n: integer

    Returns:
        Quantifier, with all_but_n_ver(_, n) as its verifier
    """
    return Quantifier("all but {}".format(n),
            isom=True, cons=True, lcons=False, rmon=None, lmon=None,
            fn=lambda seq: all_but_n_ver(seq, n))


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


def equal_number_ver(seq):
    """Generates a Quantifier corresponding to
    `the number of As equals the number of Bs'.

    Args:
        seq: sequence of elts of R^4

    Returns:
        Quantifier.T iff the number of Quantifier.ABs plus Quantifier.AnotBs is
        the same as the number of Quanitifer.BnotAs plus Quantifier.ABs
    """
    num_AB, num_AnotB, num_BnotA = 0, 0, 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
        elif np.array_equal(item, Quantifier.AnotB):
            num_AnotB += 1
        elif np.array_equal(item, Quantifier.BnotA):
            num_BnotA += 1
    return Quantifier.T if num_AnotB == num_BnotA else Quantifier.F


equal_number = Quantifier("equal number",
        isom=True, cons=False, lcons=False, rmon=None, lmon=None,
        fn=equal_number_ver)


def or_ver(q1, q2, seq):

    if q1(seq) == Quantifier.T or q2(seq) == Quantifier.T:
        return Quantifier.T
    return Quantifier.F


def at_least_n_or_at_most_m(n, m):
    return Quantifier("at_least_{}_or_at_most_{}".format(n, m),
            isom=True, cons=True, lcons=False, rmon=False, lmon=False,
            fn = lambda seq: or_ver(
                lambda seq: at_least_n_ver(seq, n),
                lambda seq: at_most_n_ver(seq, m), seq))


def get_all_quantifiers():
    """Returns: a list of all Quantifiers that have been created so far.
    """
    return [quant for quant in gc.get_objects()
            if isinstance(quant, Quantifier)]


def get_nonparity_quantifiers():

    quants = get_all_quantifiers()
    quants.remove(even)
    quants.remove(odd)
    return quants
