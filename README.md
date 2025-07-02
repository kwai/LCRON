# <img src="./images/lcron.png" width="60" height="60" style="vertical-align: middle;"> <span style="font-size: 40px; vertical-align: middle;">LCRON: Learning Cascade Ranking as One Network</span>

[![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-green)](https://github.com/RecFlow-nips24/RecFlow-nips24/blob/main/LICENSE)

---

## 📘 Overview

Cascade ranking is a widely adopted architecture in large-scale top-k selection systems, such as recommendation and advertising platforms. Traditional training methods typically optimize each stage independently, ignoring inter-stage interactions. Although recent works like RankFlow and FS-LTR attempt to model these interactions, they still face two key limitations:
1) Misalignment between per-stage objectives and the global goal of end-to-end recall.
2) Inability to learn effective collaboration patterns across stages.

To address these issues, we propose **LCRON** (**L**earning **C**ascade **R**anking as **O**ne **N**etwork), a novel framework that enables end-to-end training of multi-stage cascade ranking systems through a unified loss function.

Our method introduces a surrogate loss derived from the lower bound of the survival probability of ground-truth items through the entire cascade. Guided by the theoretical properties of this bound, we further design an auxiliary loss for each stage to tighten the bound and provide stronger supervision. This leads to more robust and effective top-k selection.

Experimental results on both public benchmarks (RecFlow) and industrial datasets demonstrate that LCRON significantly outperforms existing methods, offering substantial improvements in end-to-end Recall, system efficiency, and overall performance.

---

## 🔁 How to Reproduce Experiments on RecFlow

### 📁 Data Preparation

To run the experiments, please download the dataset from [this link](https://rec.ustc.edu.cn/share/883adf20-7e44-11ef-90e2-9beaf2bdc778). After downloading, organize the data under the `./data/` directory according to the expected structure.

---

### 🧪 Code Execution

Below are example commands to reproduce experiments for both **two-stage** and **three-stage** cascade ranking settings.

#### Two-Stage Cascade Ranking

```bash
# BCE baseline
bash ./two_stage/bce.sh

# ICC baseline
bash ./two_stage/icc.sh

# RankFlow baseline
bash ./two_stage/rankflow.sh

# FS-RankNet baseline
bash ./two_stage/fs_ranknet.sh

# FS-LambdaLoss baseline
bash ./two_stage/fs_lambdaloss.sh

# ARF baseline
bash ./two_stage/arf.sh

# ARF-v2 baseline
bash ./two_stage/arf_v2.sh

# LCRON (our method)
bash ./two_stage/lcron.sh
```


#### Two-Stage Cascade Ranking

```bash
# BCE baseline
bash ./three_stage/bce.sh

# ICC baseline
bash ./three_stage/icc.sh

# RankFlow baseline
bash ./three_stage/rankflow.sh

# FS-RankNet baseline
bash ./three_stage/fs_ranknet.sh

# FS-LambdaLoss baseline
bash ./three_stage/fs_lambdaloss.sh

# ARF baseline
bash ./three_stage/arf.sh

# ARF-v2 baseline
bash ./three_stage/arf_v2.sh

# LCRON (our method)
bash ./three_stage/lcron.sh
```

## 🛠️ How to Implement a New Method Based on the Pipeline of the Two-stage and Three-stage Cascade Ranking

Step 1. Add a new loss file under ```./deep_components/loss/two_stage``` and ```./deep_components/loss/three_stage``` You can refer to any of the files in these folders; the loss function interface can be customized.

Step 2. Add a branch: ```elif args.loss_type == "Your New Type":``` in ```./deep_components/run_train2.py``` and ```./deep_components/run_train3.py```.

Step 3. Add a new bash script under the appropriate folder (```./two_stage/``` and ```./three_stage/```).

## ⚙️ Requirements
The code has been tested under the following environment:

```yaml
python=3.7
numpy=1.18.1
pandas=1.3.5
pyarrow=8.0.0
scikit-learn=1.0.2
torch=1.13.1
faiss-gpu=1.7.1
```

> **Note:** The experiments for the three-stage cascade ranking require approximately 60GB of GPU memory. We recommend running these experiments on A800 GPUs.

## 📎 Cite this Work

If you find this project helpful for your research, please cite our paper:

```
@article{DBLP:journals/corr/abs-2503-09492,
  author       = {Yunli Wang and
                  Zhen Zhang and
                  Zhiqiang Wang and
                  Zixuan Yang and
                  Yu Li and
                  Jian Yang and
                  Shiyang Wen and
                  Peng Jiang and
                  Kun Gai},
  title        = {Learning Cascade Ranking as One Network},
  journal      = {CoRR},
  volume       = {abs/2503.09492},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2503.09492},
  doi          = {10.48550/ARXIV.2503.09492},
  eprinttype    = {arXiv},
  eprint       = {2503.09492},
  timestamp    = {Mon, 14 Apr 2025 08:08:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2503-09492.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```