# Towards Faster Graph Partitioning via Pre-training and Inductive Inference

### Abstract
Graph partitioning (GP) is a classic problem that divides the node set of a graph into densely-connected blocks. Following the IEEE HPEC Graph Challenge and recent advances in pre-training techniques (e.g., large-language models), we propose PR-GPT (Pre-trained & Refined Graph ParTitioning) based on a novel pre-training & refinement paradigm. We first conduct the *offline pre-training* of a deep graph learning (DGL) model on small synthetic graphs with various topology properties. By using the inductive inference of DGL, one can directly *generalize* the pre-trained model (with frozen model parameters) to large graphs and derive feasible GP results. We also use the derived partition as a good initialization of an efficient GP method (e.g., InfoMap) to further *refine* the quality of partitioning. In this setting, the *online generalization* and *refinement* of PR-GPT can not only benefit from the transfer ability regarding quality but also ensure high inference efficiency without re-training. Based on a mechanism of reducing the scale of a graph to be processed by the refinement method, PR-GPT also has the potential to support streaming GP. Experiments on the Graph Challenge benchmark demonstrate that PR-GPT can ensure faster GP on large-scale graphs without significant quality degradation, compared with running a refinement method from scratch.

### Citing
If you find this project useful for your research, please cite the following paper.

TBD

### Requirements
* Numpy
* Scipy
* PyTorch