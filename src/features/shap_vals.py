import logging
import os
from time import strftime

import numpy as np
import pandas as pd
import shap as shap
import torch
import transformers
from textattack.models.helpers import LSTMForClassification



logging.root.handlers = []
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/log_{}.log'.format(strftime('%d-%m-%Y-%T'))),
        logging.StreamHandler()
    ]
)

logging.getLogger().setLevel(logging.INFO)


def create_SHAP_signatures(model_name, samples, dset_name, save_name=None):

    org_text = samples['original_text']
    adv_text = samples['perturbed_text']
    num_classes = 4 if dset_name == 'ag_news' else 2 #len(samples.ground_truth_output.unique())

    if model_name.startswith('lstm'):
        model = LSTMForClassification.from_pretrained(model_name)
        tokenizer = model.tokenizer
        masker = shap.maskers.Text(r"\W")

        def f(x):
            tv = torch.tensor(
                [tokenizer.encode(v) for v in x]).cuda()
            model.cuda()
            outputs = model(tv).detach().cpu().numpy()  # Remove [0] for LSTM
            out = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
            #val = sp.special.logit(scores)
            return out
    elif model_name.startswith('textattack'):   # Huggingface transformer
        def f(x):
            tv = torch.tensor(
                [tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).cuda()
            outputs = model(tv)[0].detach().cpu().numpy()
            return (np.exp(outputs).T / np.exp(outputs).sum(-1)).T

        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        masker = tokenizer
        # Indexing error is thrown at shap: _partition explain_row(). Seems like this does not matter
    else:
        raise NotImplementedError

    explainer = shap.Explainer(f, masker)
    shap_values_org = explainer(org_text.to_list(), batch_size=20)
    shap_values_adv = explainer(adv_text.to_list(), batch_size=20)

    common_len = 512
    shap_vals_org = np.array([pad_seq(x, pad_len=common_len) for x in shap_values_org.values]).reshape(-1, num_classes * common_len)
    shap_vals_adv = np.array([pad_seq(x, pad_len=common_len) for x in shap_values_adv.values]).reshape(-1, num_classes * common_len)

    logger.info(f'Created {len(shap_vals_org)} original SHAP values with shape {shap_vals_org.shape} for {dset_name}')
    logger.info(f'Created {len(shap_vals_adv)} adversarial SHAP values with shape {shap_vals_adv.shape} for {dset_name}')


    org_save_path = f'data/SHAP_signatures/normal/{save_name}_org.npy'
    np.save(org_save_path, shap_vals_org)
    logger.info(f'Successfully saved original SHAP values to {org_save_path}')

    adv_save_path = f'data/SHAP_signatures/adversarial/{save_name}_adv.npy'
    np.save(adv_save_path, shap_vals_adv)
    logger.info(f'Successfully saved adversarial SHAP values to {adv_save_path}')


def parse_csv(csv_path):
    project_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(os.path.join(project_dir, csv_path))

    # Only return texts corresponding to successfull attacks
    df = df.loc[df['result_type'] == 'Successful']
    return df

def pad_seq(seq, pad_len=512, pad_token=0):
    return np.pad(seq[:pad_len], ((0, pad_len - seq[:pad_len].shape[0]), (0, 0)), 'constant',
                  constant_values=pad_token)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    from pathlib import Path
    Path('data/SHAP_signatures/normal/').mkdir(parents=True, exist_ok=True)
    Path('data/SHAP_signatures/adversarial/').mkdir(parents=True, exist_ok=True)

    logger.info('BAE + LSTM + AG_NEWS')
    samples = parse_csv('data/adversarial_samples/lstm_bae_ag-news.csv')
    create_SHAP_signatures('lstm-ag-news', samples, 'ag_news', 'lstm_bae_agnews')

    # LSTM + IMDB
    logger.info('BAE + LSTM + IMDB')
    samples = parse_csv('data/adversarial_samples/lstm_bae_imdb.csv')
    create_SHAP_signatures('lstm-imdb', samples, 'imdb', 'lstm_bae_imdb')

    # LSTM + SST-2
    logger.info('BAE + LSTM + SST-2')
    samples = parse_csv('data/adversarial_samples/lstm_bae_SST-2.csv')
    create_SHAP_signatures('lstm-sst2', samples, 'sst2', 'lstm_bae_sst2')

