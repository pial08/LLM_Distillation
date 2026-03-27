# Clone What You Can’t Steal: Black-Box LLM Replication via Logit Leakage and Distillation

This repository provides a similar prototype implementation of the methods described in our paper:

> Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz. (2025). **Clone what you can’t steal: Black-box LLM replication via logit leakage and distillation.** *Proceedings of the 7th IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS).* Advance online publication. https://arxiv.org/abs/2509.00973 

The codebase demonstrates the overall pipeline, structure, and experimental workflow used in the study, including projection-matrix recovery, knowledge distillation, and evaluation. This is not the exact original research code; instead, it is a cleaned, representative implementation intended for transparency and reproducibility.



## Overview

The project studies black-box model replication of large language models (LLMs) under constrained access, assuming only top-k logit exposure via an API.

The pipeline consists of two stages:

1. Stealing stage – Recovering the output projection subspace using SVD over leaked logits.
2. Cloning stage – Distilling the remaining transformer behavior into compact student models.



## Repository Structure

```text
LLM_Smashdown/
├── LLM_smashdown.py          # Main training entry (local): projection recovery + distillation
├── LLM_smashdown_HPC.py      # Training entry (HPC/cluster): scheduler/multi-GPU friendly

├── config.json               # Experiment configuration (models, loss, data paths, hyperparams)
├── requirements.txt          # Python dependencies

├── EDA.ipynb                 # Data sanity checks + exploratory analysis
├── Evaluation.ipynb          # Core evaluation: PPL/NLL/KL + token alignment + similarity
├── Evaluation_2.ipynb        # Extended eval: generalization + AIC/AICc + efficiency trade-offs

├── Evaluation Results/       # Saved figures + intermediate evaluation artifacts
├── results/                  # Final outputs: logs, metrics summaries, exported results

├── Directory_Structure.txt   # Expected folder layout / path conventions
```


## Running the Code

```shell
# Install dependencies:
pip install -r requirements.txt

# Local training:
python LLM_smashdown.py

# HPC / cluster training:
python LLM_smashdown_HPC.py
```


## Evaluation

All evaluation is performed using Jupyter notebooks:
- `EDA.ipynb` for dataset understanding
- `Evaluation.ipynb` for core metrics
- `Evaluation_2.ipynb` for advanced analysis

Results are saved under `Evaluation Results/` and `results/`.



## Notes

- No proprietary models, weights, or APIs are included.
- All experiments rely on public datasets and simulated black-box access.
- This repository is intended for research, red-team simulation, and defensive analysis.




## 📖 Citation

If you use this repository for your research, please cite our paper accepted at the **7th IEEE International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS)**:

*Kanchon Gharami*, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation. arXiv preprint arXiv:2509.00973. Accepted, in publication. (2025)

**BibTeX:**
```bibtex
@inproceedings{gharami2025tps,
  author    = {Gharami, Kanchon and Aluvihare, Hansaka and Moni, Shafika Showkat and Peköz, Berker},
  title     = {Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation},
  booktitle = {Proceedings of the 7th IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS)},
  year      = {2025},
  note      = {Accepted, in publication},
}
```
or,

```bibtex
@article{gharami2025clone,
  title    = {Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation},
  author   = {Gharami, Kanchon and Aluvihare, Hansaka and Moni, Shafika Showkat and Peköz, Berker},
  journal  = {arXiv preprint arXiv:2509.00973},
  year     = {2025},
  url      = {https://arxiv.org/abs/2509.00973}
}
```


## Authors

**Kanchon Gharami** – Ph.D. Student, Department of Electrical Engineering and Computer Science  
**Hansaka Aluvihare** – M.S. Graduate, Department of Mathematics  
**Shafika Showkat Moni** – Assistant Professor, Department of Electrical Engineering and Computer Science  
**Berker Peköz** – Assistant Professor, Department of Electrical Engineering and Computer Science  

Embry-Riddle Aeronautical University, Daytona Beach, FL, USA


## Contact
For questions or issues, please contact gharamik@my.erau.edu or kanchon2199@gmail.com

