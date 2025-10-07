# Causal Abstraction for Mechanistic Interpretability

This repository supports mechanistic interpretability experiments that reverse engineer what algorithm a neural network implements with causal abstraction.

 It supports the baseline experiments for the causal variable localization track of the Mechanistic Interpretability Benchmark (MIB).
[![Paper](https://img.shields.io/badge/MIB_Paper-arXiv-b31b1b)](https://arxiv.org/abs/mib-paper) 
[![GitHub](https://img.shields.io/badge/GitHub-MIB-blue)](https://github.com/mechanistic-interpretability-benchmark/mib)

## Overview

 This codebase follows a causal abstraction approach, where we hypothesize high-level causal models of how LLMs might solve tasks, and then locate where and how these abstract variables are represented in the model.

### Causal Models
A causal model (causal_model.py) consists of:

- **Variables**: Concepts that might be represented in the neural network
- **Values**: Possible assignments to each variable
- **Parent-Child Relationships**: Directed relationships showing causal dependencies
- **Mechanisms**: Functions that compute a variable's value given its parents' values

### Causal Abstraction

Mechanistic interpretability aims to reverse-engineer what algorithm a neural network implements to achieve a particular capability. Causal abstraction is a theoretical framework that grounds out these notions; an algorithm is a causal model, a neural network is a causal model, and the notion of implementation is the relation of causal abstraction between two models. The algorithm is a **high-level causal model** and the neural network is a **low-level causal model**.  When the high-level mechanisms are accurate simplifications of the low-level mechanisms, the algorithm is a **causal abstraction** of the low-level causal model.

### Neural Network Features

What are the basic building blocks we should look at when trying to understand how AI systems work internally? This question is still being debated among researchers. 
The causal abstraction framework remains agnostic to this question by allowing for building blocks of any shape and sizes that we call **features**. The features of a hidden vector in a neural network are accessed via an invertible **featurizer**, which might be an orthogonal matrix, the identity function, or an autoencoder. The neural network components are implemented in the `neural/` directory with modular access to different model units.


### Interchange Interventions
We use interchange interventions to test if a variable in a high-level causal model aligns with specific features in the LLM. An interchange intervention replaces values from one input with values from another input, allowing us to isolate and test specific causal pathways.


The codebase implements five baseline approaches for feature construction and selection:

1. **Full Vector**: Uses the entire hidden vector without any transformations.

2. **DAS (Distributed Alignment Search)**: Learns orthogonal directions with supervision from the causal model.

3. **DBM (Desiderata-Based Masking)**: Learns binary masks over features using the causal model for supervision. Can be applied to select neurons (standard dimensions of hidden vectors), PCA components, or SAE features.

4. **PCA (Principal Component Analysis)**: Uses unsupervised orthogonal directions derived from principal components. DBM can be used to align principal components with a high-level causal variable.

5. **SAE (Sparse Autoencoder)**: Leverages pre-trained sparse autoencoders like GemmaScope and LlamaScope. DBM can be used to align principal components with a high-level causal variable.

## Repository Structure

### Core Components

#### `causal/`
- `causal_model.py`: Implementation of causal models with variables, values, parent-child relationships, and mechanisms for counterfactual generation and intervention mechanics
- `counterfactual_dataset.py`: Dataset handling for counterfactual data generation and management

#### `neural/`
- `pipeline.py`: Abstract base pipeline and LM pipeline classes for consistent interaction with different language models
- `model_units.py`: Base classes for accessing model components and features in transformer architectures
- `LM_units.py`: Language model specific components for residual stream and attention head access
- `featurizers.py`: Invertible feature space definitions with forward/inverse featurizer modules and intervention utilities

#### `experiments/`
- `pyvene_core.py`: Core utilities for creating, managing, and running intervention experiments using the pyvene library
- `attention_head_experiment.py`: Experiments targeting attention head components
- `residual_stream_experiment.py`: Experiments on residual stream representations
- `intervention_experiment.py`: General intervention experiment framework
- `filter_experiment.py`: Filtering and selection experiments
- `aggregate_experiments.py`: Tools for running and aggregating multiple experiments

#### `tests/`
Comprehensive test suite covering all core components with specialized tests for pyvene integration in `test_pyvene_core/`

## Getting Started

### Installation

```bash
git clone https://github.com/your-org/causal-abstraction.git
cd causal-abstraction
pip install -r requirements.txt
```

### Key Dependencies

- **PyTorch**: Deep learning framework for model operations
- **pyvene**: Library for causal interventions on neural networks
- **transformers**: Hugging Face library for language model access
- **pytest**: Testing framework