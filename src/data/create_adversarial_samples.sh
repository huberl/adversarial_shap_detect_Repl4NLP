#---------------------------- PWWS ----------------------------
## LSTM
textattack attack --recipe pwws --model lstm-ag-news --dataset-from-huggingface ag_news --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_pwws_ag-news.csv --num-workers-per-device 8
textattack attack --recipe pwws --model lstm-imdb --dataset-from-huggingface imdb --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_pwws_imdb.csv --num-workers-per-device 8
textattack attack --recipe pwws --model lstm-sst2 --dataset-from-huggingface sst2 --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_pwws_SST-2.csv --num-workers-per-device 8

## DistilBERT
textattack attack --recipe pwws --model distilbert-base-uncased-ag-news --dataset-from-huggingface ag_news --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_pwws_ag-news.csv --num-workers-per-device 8
textattack attack --recipe pwws --model distilbert-base-uncased-imdb --dataset-from-huggingface imdb --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_pwws_imdb.csv --num-workers-per-device 8
textattack attack --recipe pwws --model distilbert-base-cased-sst2 --dataset-from-huggingface sst2 --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_pwws_SST-2.csv --num-workers-per-device 8

# Roberta
textattack attack --recipe pwws --model roberta-base-ag-news --dataset-from-huggingface ag_news --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/roberta_pwws_ag-news.csv --num-workers-per-device 8
textattack attack --recipe pwws --model roberta-base-imdb --dataset-from-huggingface imdb --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/roberta_pwws_imdb.csv --num-workers-per-device 8
textattack attack --recipe pwws --model roberta-base-SST-2 --dataset-from-huggingface sst2 --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/roberta_pwws_SST-2.csv --num-workers-per-device 8


#---------------------------- BAE ----------------------------
## LSTM
textattack attack --recipe bae --model lstm-ag-news --dataset-from-huggingface ag_news --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_bae_ag-news.csv --num-workers-per-device 8
textattack attack --recipe bae --model lstm-imdb --dataset-from-huggingface imdb --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_bae_imdb.csv --num-workers-per-device 8
textattack attack --recipe bae --model lstm-sst2 --dataset-from-huggingface sst2 --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/lstm_bae_SST-2.csv --num-workers-per-device 8

## DistilBERT
textattack attack --recipe bae --model distilbert-base-uncased-ag-news --dataset-from-huggingface ag_news --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_bae_ag-news.csv --num-workers-per-device 8
textattack attack --recipe bae --model distilbert-base-uncased-imdb --dataset-from-huggingface imdb --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_bae_imdb.csv --num-workers-per-device 8
textattack attack --recipe bae --model distilbert-base-cased-sst2 --dataset-from-huggingface sst2 --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/distilbert_bae_SST-2.csv --num-workers-per-device 8

# Roberta
textattack attack --recipe bae --model roberta-base-ag-news --dataset-from-huggingface ag_news --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/roberta_bae_ag-news.csv --num-workers-per-device 8
textattack attack --recipe bae --model roberta-base-imdb --dataset-from-huggingface imdb --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/roberta_bae_imdb.csv --num-workers-per-device 8
textattack attack --recipe bae --model roberta-base-SST-2 --dataset-from-huggingface sst2 --csv-coloring-style plain --num-successful-examples 2000 --random-seed 0 --log-to-csv data/adversarial_samples/roberta_bae_SST-2.csv --num-workers-per-device 8



textattack attack --recipe pwws --model-from-huggingface distilbert-base-uncased-ag-news
textattack attack --recipe pwws --model distilbert-base-uncased-imdb --dataset-from-huggingface imdb
textattack attack --recipe pwws --model distilbert-base-cased-sst2 --dataset-from-huggingface sst2
