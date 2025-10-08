# Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context
This repository contains the official code of the paper: "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context" ([link](https://arxiv.org/abs/2510.06182)).

<p align="center">
  <img width="864" height="830" alt="mechs_fig1" src="https://github.com/user-attachments/assets/e3ac9cdf-add7-4f02-96d0-f2b75e359651" />
</p>

### Files
The codebase is still being finalized, but for now I uploaded the main files used in the paper. These are:
- `CausalAbstraction/` - this is a copy of the official [CausalAbstraction](https://github.com/atticusg/CausalAbstraction) codebase, with very minor quality of life tweaks that I should open a PR for. For now, for the sake of reproducibility, I just put my version of the code here.
- `grammar/` - this directory includes files that both define all of our binding tasks (`schemas.py`), as well as code that automatically turns them into a CausalModel that can be used with the `CausalAbstraction` codebase (`task_to_causal_model.py`).
- `tasks/dist.py` - this file includes the code for running most of our experiments. You can pick a counterfactual, which model to run on, etc.
- `training.py` - code containing lots of counterfactuals and other setup code needed by `dist.py`.
- `plotting.py` - code for generating our main figure.
- `example.ipynb` - an example script that should just work out of the box, running our main interchange intervention and plotting the results.

---

### Citation
Please cite as:
```
@misc{gurarieh2025mixing,
    title={Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context},
    author={Yoav Gur-Arieh and Mor Geva and Atticus Geiger},
    year={2025},
    eprint={2510.06182},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---

Please check out our [interactive blog post](https://yoav.ml/blog/2025/mixing-mechs/) to explore how the different binding mechanisms affect model behavior!

<p align="center">
<img width="800" height="905" alt="mechs_fig1" src="https://github.com/user-attachments/assets/4c028f8f-c83c-43c1-aa85-f1a0aef92333" />
</p>


Feel free to contact if you have any thoughts, questions or suggestions.
