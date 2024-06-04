# train bi-encoder
python3 train_bi-encoder_mnrl.py

# evaluate bi-encoder
python3 eval_msmarco.py models/multi-qa-distilbert-cos-v1 50

# evaluate cross-encoder
python3 eval_cross-encoder-trec-dl.py models/qnli-distilroberta-base
