# Knowledge Graph Attention Network
This is PyTorch & DGL implementation for the paper:
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

You can find Tensorflow implementation by the paper authors [here](https://github.com/xiangwang1223/knowledge_graph_attention_network).

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.

If you want to use codes and datasets in your research, please contact the paper authors and cite the following paper as the reference:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
* torch == 1.3.1
* dgl-cu90 == 0.4.1
* numpy == 1.15.4
* pandas == 0.23.1
* scipy == 1.1.0
* sklearn == 0.20.0

## Run the Codes
* FM
```
python main_nfm.py --model_type fm --data_name amazon-book
```
* NFM
```
python main_nfm.py --model_type nfm --data_name amazon-book
```
* BPRMF *(train on multi-GPUs)*
```
python -m torch.distributed.launch main_bprmf.py --data_name amazon-book
```
* ECFKG *(train on multi-GPUs)*
```
python -m torch.distributed.launch main_ecfkg.py --data_name amazon-book
```
* CKE *(train on multi-GPUs)*
```
python -m torch.distributed.launch main_cke.py --data_name amazon-book
```
* KGAT
```
python main_kgat.py --data_name amazon-book
```

## Results
With my code, following are the results of each model when training with dataset `amazon-book`.

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| FM    | sample 1000 test users | 65         | 0.014400000683963299 | 0.14490722119808197 | 0.07221827559341328 |
| NFM   | sample 1000 test users | 56         | 0.013850000686943531 | 0.13833996653556824 | 0.0724611583347469  |
| BPRMF | all test users         | 65         | 0.014154779163154574 | 0.13356850621872207 | 0.06943918307731874 |
| ECFKG | all test users         | 41         | 0.013035656309061863 | 0.12247500353257905 | 0.06115661206228789 |
| CKE   | all test users         | 52         | 0.014507515353912879 | 0.13836056015380443 | 0.07225836488142431 |
| KGAT  | all test users         | 31         | 0.014817044902584718 | 0.14117674635791852 | 0.07526633940808744 |

Final results on all test users

| Model | Precision@20 | Recall@20 | NDCG@20 |
| :---: | :---:        | :---:     | :---:   |
| FM    | 0.0138       | 0.1309    | 0.0676  |
| NFM   | 0.0131       | 0.1246    | 0.0655  |
| BPRMF | 0.0142       | 0.1336    | 0.0694  |
| ECFKG | 0.0130       | 0.1225    | 0.0612  |
| CKE   | 0.0145       | 0.1384    | 0.0723  |
| KGAT  | 0.0148       | 0.1412    | 0.0753  |

## Related Papers
* FM
    * Proposed in [Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002), SIGIR2011.

* NFM
    * Proposed in [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777), SIGIR2017.

* BPRMF
    * Proposed in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=1795167), UAI2009.
    * Key point: 
        * Replace point-wise with pair-wise.

* ECFKG
    * Proposed in [Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation](https://arxiv.org/abs/1805.03352), Algorithm2018.
    * Implementation by the paper authors: [https://github.com/evison/KBE4ExplainableRecommendation](https://github.com/evison/KBE4ExplainableRecommendation)
    * Key point: 
        * Introduce Knowledge Graph to Collaborative Filtering

* CKE
    * Proposed in [Collaborative Knowledge Base Embedding for Recommender Systems](https://dl.acm.org/citation.cfm?id=2939673), KDD2016.
    * Key point: 
        * Leveraging structural content, textual content and visual content from the knowledge base.
        * Use TransR which is an approach for heterogeneous network, to represent entities and relations in distinct semantic space bridged by relation-specific  matrices.
        * Performing knowledge base embedding and collaborative filtering jointly.

* KGAT
    * Proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD2019.
    * Implementation by the paper authors: [https://github.com/xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)
    * Key point:
        * Model the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.
        * Train KG part and CF part in turns.
        


