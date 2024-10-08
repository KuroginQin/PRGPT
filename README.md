# Towards Faster Graph Partitioning via Pre-training and Inductive Inference

This repository provides a reference implementation of PR-GPT introduced in the paper "**Towards Faster Graph Partitioning via Pre-training and Inductive Inference**", which won the [**Champion of IEEE HPEC 2024 Graph Challenge**](https://graphchallenge.mit.edu/champions).

### Abstract
Graph partitioning (GP) is a classic problem that divides the node set of a graph into densely-connected blocks. Following the IEEE HPEC Graph Challenge and recent advances in pre-training techniques (e.g., large-language models), we propose PR-GPT (Pre-trained & Refined Graph ParTitioning) based on a novel pre-training & refinement paradigm. We first conduct the *offline pre-training* of a deep graph learning (DGL) model on small synthetic graphs with various topology properties. By using the inductive inference of DGL, one can directly *generalize* the pre-trained model (with frozen model parameters) to large graphs and derive feasible GP results. We also use the derived partition as a good initialization of an efficient GP method (e.g., InfoMap) to further *refine* the quality of partitioning. In this setting, the *online generalization* and *refinement* of PR-GPT can not only benefit from the transfer ability regarding quality but also ensure high inference efficiency without re-training. Based on a mechanism of reducing the scale of a graph to be processed by the refinement method, PR-GPT also has the potential to support streaming GP. Experiments on the Graph Challenge benchmark demonstrate that PR-GPT can ensure faster GP on large-scale graphs without significant quality degradation, compared with running a refinement method from scratch.

### Citing
If you find this project useful for your research, please cite the following paper.

```
@article{qin2024towards,
  title={Towards Faster Graph Partitioning via Pre-training and Inductive Inference},
  author={Qin, Meng and Zhang, Chaorui and Gao, Yu and Ding, Yibin and Jiang, Weipeng and Zhang, Weixi and Han, Wei and Bai, Bo},
  journal={arXiv preprint arXiv:2409.00670},
  year={2024}
}
```
```
@inproceedings{qin2024pre,
  title={Pre-train and refine: Towards higher efficiency in k-agnostic community detection without quality degradation},
  author={Qin, Meng and Zhang, Chaorui and Gao, Yu and Zhang, Weixi and Yeung, Dit-Yan},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2467--2478},
  year={2024}
}
```

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
* numpy
* scipy
* pytorch
* infomap
* sdp-clustering
* graph-tool

### Usage

To run *InfoMap* (from scratch) on static GP with a specific setting of N:
~~~
python InfoMap_static.py --N 100000
~~~
To run *InfoMap* (from scratch) on streaming GP with a specific setting of N & ind-th graph (ind=[1,2,3,4,5])
~~~
python InfoMap_stream.py --N 100000 --ind 1
~~~

To run *Locale* (from scratch) on static GP with a specific setting of N:
~~~
python Locale_static.py --N 100000
~~~
To run *Locale* (from scratch) on streaming GP with a specific setting of N & ind-th graph (ind=[1,2,3,4,5])
~~~
python Locale_stream.py --N 100000 --ind 1
~~~

To run PR-GPT on static GP w/ a specific setting of N using a pre-trained model after eph-th epochs:
~~~
python PRGPT_static.py --N 10000 --eph 4
python PRGPT_static.py --N 50000 --eph 4
python PRGPT_static.py --N 100000 --eph 4
python PRGPT_static.py --N 500000 --eph 5
python PRGPT_static.py --N 1000000 --eph 8
~~~
To run PR-GPT on streaming GP w/ a specific setting of N & ind-th graph (ind=[1,2,3,4,5]) using a pre-trained model after eph-th epochs:
~~~
python PRGPT_stream.py --N 100000 --eph 4 --ind 1
~~~

To conduct the offline pre-training of PR-GPT, download our [generated (synthetic) pre-training data](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mqinae_connect_ust_hk/EU29ZyTZA6VChRakolwFHvoBO7qdjTHJFatl0056p8-bbA?e=CUAMD7), unzip the file, and put the derived files in ./data. Then:
~~~
python PRGPT_ptn.py
~~~
After the offline pre-training, please copy the saved model parameters (e.g., in ./chpt_new) to ./chpt.

To generate new pre-training data (i.e., small synthetic graphs):
~~~
python ptn_data_gen.py
~~~

Please note that different environment setups (e.g., CPU, GPU, memory size, versions of libraries and packages, etc.) may result in different evaluation results regarding the inference time. When testing the inference time, please also make sure that there are no other processes with heavy resource requirements (e.g., GPUs and memory) running on the same server. Otherwise, the evaluated inference time may not be stable.
