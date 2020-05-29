#### TP-N2F encoder-decoder model.
This is an implementation of TP-N2F, a new encoder-decoder model based
on Tensor Product Representations (TPRs) for Natural- to Formal-language
generation proposed in
[Natural- to formal-language generation using Tensor Product Representations](https://arxiv.org/abs/1910.02339).

Except for the obviously necessary substitution of argmax for softargmax
in the unbinding submodule and use of Xavier uniform initializer here
and there, the implementation follows the paper almost to the letter. It
comes with out-of-the-box support for the
[MathQA](https://math-qa.github.io/math-QA/data/MathQA.zip) dataset and
for running experiments on it with all the default hyper parameters
described in the article, though of the two suggested metrics this
implementation counts operation accuracy only.

On top of that, in contrast to what the authors proposed, this
implementation is not confined to only binary or ternary relations and
works with relations of arbitrary arity.  
That means the MathQA dataset (or any other dataset with formulae) can
be used as-is without jugglery of operations unlike what the the paper
promotes.  
Also all the dataset related hyper parameters (such as the size of the
argument vocabulary, the size of the relation vocabulary, its size and
the maximum of the arities) are automatically calculated from the
dataset.  
With all that the implementation easily adapts to other MathQA-like
datasets.

#### Setting things up.
Change to the project directory and run `pip install .` from the command
line.  
Then run `python train.py` to train the model with the editorial hyper
parameters.  
Type `python train.py --help` to scope out ingredients for crafting your
own experiments.  
To infer solution to your problem run `python infer.py` for directions,
model checkpoints can be usually found in the `lightning_logs` directory
after a model has been trained.  
You can also play with the notebook after running `jupyter notebook
tpn2f_demo.ipynb`.
