# GraphPrompter: Multi-stage Adaptive Prompt Optimization for Graph In-Context Learning

A multi-stage adaptive graph prompt selection algotithm to enhance the in-context learning over graphs.

<!-- ![GraphPrompter](overview-of-GraphPrompter.png) -->


# Setup

```
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

```

All datasets should be prepared to individual folders under <DATA_ROOT>. For MAG and arXiv, the datasets will be automatically downloaded and processed to <DATA_ROOT>. In case of memory issue when generating adjacency matrix, we also provide the preprocessed MAG [adjacency matrix](http://snap.stanford.edu/prodigy/mag240m_adj_bi.pt) that should be put under <DATA_ROOT>/mag240m after the ogb download.

For KG, download preprocessed [Wiki](http://snap.stanford.edu/prodigy/Wiki.zip) and [FB15K-237](http://snap.stanford.edu/prodigy/FB15K-237.zip) datasets to <DATA_ROOT>. Download other KG datasets ([NELL](http://snap.stanford.edu/csr/NELL.zip) and [ConceptNet](http://snap.stanford.edu/csr/ConceptNet.zip)) similarly following links in https://github.com/snap-stanford/csr.

# Pretraining and Evaluation

pretrain on mag240m : `pretrain_mag240m.sh`

pretrain on wiki : `pretrain_wiki.sh`

> We provide pretrained model in folder `./state`, you can use these models straightly.

evaluation on arxiv: `eval_arxiv.sh`

evaluation on ConcepNet/FB15K-237/NELL: `eval_all_kg.sh`


# Citations

If you use this repo, please cite the following paper. This repo reuses code from [CSR](https://github.com/snap-stanford/csr) for KG datasets loading.

article(Under review):  GraphPrompter: Multi-stage Adaptive Prompt Optimization for Graph In-Context Learning
