# 🍝 PASTA | Adaptive Smooth Tchebycheff Attention for Multi-Objective Policy Optimization

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://alejandromllo.github.io/research/pasta/)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](https://arxiv.org/abs/2605.12771)
<!-- [![Conference](https://img.shields.io/badge/RSS-2026-brightgreen)]() -->

This repository contains the official PyTorch implementation of the **PASTA** 🍝 (**P**olicy-optimization via **A**daptive **S**mooth **T**chebycheff **A**ttention) algorithm, as presented in the paper:

> Alejandro Murillo-González, Mahmoud Ali, and Lantao Liu. **Adaptive Smooth Tchebycheff Attention for Multi-Objective Policy Optimization.** Robotics: Science and Systems (RSS) 2026

PASTA 🍝 enables stable policy optimization for non-Convex Pareto tradeoffs.

## 🔥 Quick Start

To quickly run the PASTA 🍝 algorithm and reproduce the included minimal demo run the following commands.

1. Clone the repository:
```bash
git clone https://github.com/AlejandroMllo/PASTA.git
cd PASTA
```

2. Create the Conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate pasta-env
```

4. Run the Multi-Objective demo:
```bash
python src/demo_pasta.py
```

The provided demo illustrates the PASTA 🍝 algorithm within a deliberately minimal multi-objective environment, designed to facilitate rapid evaluation and seamless adaptation.

## 🚀 What is included

This repository includes:

- `src/pasta.py` – single-file implementation of the PASTA 🍝 algorithm, including the smoothness controller.
- `src/demo_pasta.py` – a minimal demo script showing how to run PASTA 🍝 on a simple MORL environment
- `README.md` – this documentation
- `environment.yml` – Conda environment file with the required dependencies

## 📌 Recommended usage

To use the algorithm in your own research or experiments:

1. Inspect `demo_pasta.py` to see how the environment and training loop are configured.
2. Import the core PASTA 🍝 classes from the `src/pasta.py` file. 
3. Customize the actor and critic architectures, if needed, to handle your environment's observation space.

## 📚 Citation

If you use this code or the PASTA 🍝 algorithm in your work, please cite:

```
@INPROCEEDINGS{murillo2026pasta,
    AUTHOR    = {Alejandro Murillo-González AND Mahmoud Ali AND Lantao Liu},
    TITLE     = {{Adaptive Smooth Tchebycheff Attention for Multi-Objective Policy Optimization}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2026},
    ADDRESS   = {Sydney, Australia},
    MONTH     = {July}
}
```
