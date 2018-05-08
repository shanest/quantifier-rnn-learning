# Learnability and Semantic Universals

This repository accompanies the following paper:
* Steinert-Threlkeld and Szymanik 2018, "Learnability and Semantic Universals", under revision at _Semantics & Pragmatics_. https://semanticsarchive.net/Archive/mQ2Y2Y2Z/LearnabilitySemanticUniversals.pdf

In particular, we attempt to explain semantic universals surrounding quantifiers by training a recurrent neural network (a long-short term memory network specifically) to learn quantifiers, some satisfying proposed universals and some not.  Generally, we find that those satisfying universals are easier to learn by this model than those that do not.  

This repository contains all the code needed to replicate the experiments reported in that paper.  It also contains the data and figures from the experiments that we ran and which are reported therein.

## Requirements

Python 2.7, TensorFlow 1.4+, Pandas

I plan to make the code compatible with Python 3 soon.

## Running Experiments

Any of the experiments can be re-run very simply from the command-line.  For example, to run experiment 1(a):

```
python quant_verify.py --exp one_a --out_path /tmp/exp1a/
```

Note that the output written to `out_path` will also contain checkpoints and other data from TensorFlow in addition to CSV files recording the relevant information from each trial.
