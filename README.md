# Learnability and Semantic Universals

This repository accompanies the following paper:
* Steinert-Threlkeld and Szymanik 2018, "Learnability and Semantic Universals", under revision at _Semantics & Pragmatics_. https://semanticsarchive.net/Archive/mQ2Y2Y2Z/LearnabilitySemanticUniversals.pdf

In particular, we attempt to explain semantic universals surrounding quantifiers by training a recurrent neural network (a long-short term memory network specifically) to learn quantifiers, some satisfying proposed universals and some not.  Generally, we find that those satisfying universals are easier to learn by this model than those that do not.  

This repository contains all the code needed to replicate the experiments reported in that paper.  It also contains the data and figures from the experiments that we ran and which are reported therein.

If you have any questions and/or want to extend the code-base and/or run your own experiments, feel free to get in touch!

## Requirements

Python 2.7+, TensorFlow 1.4+, Pandas

[NB:  the code should be compatible with Python 3, but has not been tested with it.]

## Running Experiments

Any of the experiments can be re-run very simply from the command-line.  For example, to run experiment 1(a):

```
python quant_verify.py --exp one_a --out_path /tmp/exp1a/
```

Note that the output written to `out_path` will also contain checkpoints and other data from TensorFlow in addition to CSV files recording the relevant information from each trial.

## Analyzing Data

To analyze the output from an experiment, use the `analysis.experiment_analysis` method.  It takes two arguments: a path to a directory containing the CSV files output by a run of an experiment, and an array containing the names of quantifiers in the experiment.

To re-create the analysis from the paper and the figures included therein, you can use the predefined methods.  For example, in a Python shell:

```
import analysis
analysis.experiment_one_a_analysis()
```

## Designing and Running New Experiments

The code is designed to make it easy to design and run new experiments.  You can look at any of the `experiment_` methods in `quant_verify.py` for inspiration.  I recommend copy/pasting one of these methods, renaming it, updating the parameters to what you would like, and then running it!  The next section tells you how to make new quantifiers if you want to test some not included.

## Defining New Quantifiers

All of the quantifiers are defined in `quantifiers.py`.  The core is a verification method, named `_ver(seq)`.  This takes in a sequence of characters (named `Quantifier.AB`, `Quantifier.AnotB`, et cetera) and outputs one of the truth values `Quantifier.T` or `Quantifier.F`.  This method implements the core semantics of a quantifier.

The verification method is then wrapped inside a `Quantifier` object, which provides a name for the quantifier and assigns it certain semantic properties.  Note that the name used here must match the names passed in as strings to the `experiment_analysis` method.

Again, I recommend using an example from `quantifiers.py` as your starting point for defining a new quantifier.
