# Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context
This repository contains the official code of the paper: "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context" (TODO add link).

### Files
The codebase is still being finalized, but for now I uploaded the main files used in the paper. These are:
- `CausalAbstraction/` - this is a copy of the official [CausalAbstraction](https://github.com/atticusg/CausalAbstraction) codebase, with very minor quality of life tweaks that I should open a PR for. For now, for the sake of reproducibility, I just put my version of the code here.
- `grammar/` - this directory includes files that both define all of our binding tasks (`schemas.py`), as well as code that automatically turns them into a CausalModel that can be used with the `CausalAbstraction` codebase (`task_to_causal_model.py`).
- `tasks/dist.py` - this file includes the code for running most of our experiments. You can pick a counterfactual, which model to run on, etc.
- `training.py` - code containing lots of counterfactuals and other setup code needed by `dist.py`.
- `plotting.py` - code for generating our main figure.
- `example.ipynb` - an example script that should just work out of the box, running our main interchange intervention and plotting the results.

### Citation
Please cite as:
TODO
<!-- ```
@article{geva2021strategyqa,
  title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  year = {2021},
}
``` -->

Feel free to contact if you have any thoughts, questions or suggestions.