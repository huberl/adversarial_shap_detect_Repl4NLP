#---------------------------- PWWS ----------------------------
## LSTM
textattack attack --recipe pwws --model lstm-ag-news --dataset-from-huggingface ag_news --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_pwws_agnews.csv
textattack attack --recipe pwws --model lstm-imdb --dataset-from-huggingface imdb --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_pwws_imdb.csv
textattack attack --recipe pwws --model lstm-sst2 --dataset-from-huggingface sst2 --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_pwws_sst2.csv

## DistilBERT
textattack attack --recipe pwws --model distilbert-base-uncased-ag-news --dataset-from-huggingface ag_news --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_pwws_agnews.csv
textattack attack --recipe pwws --model distilbert-base-uncased-imdb --dataset-from-huggingface imdb --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_pwws_imdb.csv
textattack attack --recipe pwws --model distilbert-base-cased-sst2 --dataset-from-huggingface sst2 --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_pwws_sst2.csv

#---------------------------- BAE ----------------------------
## LSTM
textattack attack --recipe bae --model lstm-ag-news --dataset-from-huggingface ag_news --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_bae_agnews.csv
textattack attack --recipe bae --model lstm-imdb --dataset-from-huggingface imdb --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_bae_imdb.csv
textattack attack --recipe bae --model lstm-sst2 --dataset-from-huggingface sst2 --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_bae_sst2.csv

## DistilBERT
textattack attack --recipe bae --model distilbert-base-uncased-ag-news --dataset-from-huggingface ag_news --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_bae_agnews.csv
textattack attack --recipe bae --model distilbert-base-uncased-imdb --dataset-from-huggingface imdb --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_bae_imdb.csv
textattack attack --recipe bae --model distilbert-base-cased-sst2 --dataset-from-huggingface sst2 --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_bae_sst2.csv
