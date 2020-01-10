python main_nfm.py
python main_kgat.py

python -m torch.distributed.launch main_bprmf.py
python -m torch.distributed.launch main_ecfkg.py
python -m torch.distributed.launch main_cke.py


(Sampled Test Data)
FM      65.0    0.014400000683963299    0.14490722119808197     0.07221827559341328
NFM     56.0    0.013850000686943531    0.13833996653556824     0.0724611583347469

(Full Test Data)
KGAT    31.0    0.014817044902584718    0.14117674635791852     0.07526633940808744
BPRMF   65.0    0.014154779163154574    0.13356850621872207     0.06943918307731874




BPRMF   : Precision 0.0142   Recall 0.1336   NDCG 0.0694
NFM     : Precision 0.0131   Recall 0.1246   NDCG 0.0655
FM      : Precision 0.0138   Recall 0.1309   NDCG 0.0676



ECFKG: https://github.com/evison/KBE4ExplainableRecommendation


