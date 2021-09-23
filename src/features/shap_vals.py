import glob
import os

import shap as shap
import textattack

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
from pathlib import Path
import re
import numpy as np
from datasets import load_dataset
import torch

from os import walk

from src.models.models import get_model_by_name
from src.models.models import evaluate_torch, evaluate_huggingface


def create_SHAP_signatures(model_name, samples, dset_name, save_path=None):
    model, tokenizer = get_model_by_name(model_name, with_wrapper=False)

    org_text = samples['original_text']
    adv_text = samples['perturbed_text']


    if isinstance(model, textattack.models.helpers.LSTMForClassification):
        acc = evaluate_torch(model, tokenizer, adv_text, samples['ground_truth_output'])

        org_text = torch.Tensor(tokenizer(samples['original_text'].to_list()))
        adv_text = torch.Tensor(tokenizer(samples['perturbed_text'].to_list()))
        dataset = textattack.datasets.HuggingFaceDataset(dset_name, split='test')

        # TODO: Why sst2?
        data = np.array(tokenizer(load_dataset('sst2', 'default')['test']['text']))
        background = data[np.random.choice(data.shape[0], 500, replace=False)]
        background = torch.tensor(background, dtype=torch.int)
        #masker = shap.maskers.Independent(data=background)
        model.to('cpu')
        explainer = shap.DeepExplainer(model, background)
    else:
        device = 0 if torch.cuda.is_available() else -1         # -1 defaults to CPU
        # We need all output scores for the SHAP calculation
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

        explainer = shap.Explainer(pipe)
        acc = evaluate_huggingface(model, tokenizer, adv_text, samples['ground_truth_output'])

    assert acc == 0.0, f'Not all samples are adversarial! Acc: {acc:.2f}'

    shap_vals_org = explainer(org_text)
    shap_vals_adv = explainer(adv_text)

    def pad_seq(seq, pad_len=128, pad_token=0, cutoff=True):

        return np.pad(seq[:pad_len], ((0, pad_len - seq[:pad_len].shape[0]), (0, 0)), 'constant', constant_values=pad_token)


    shap_vals_org = np.array([pad_seq(x) for x in shap_vals_org.values]).reshape(-1, 512)
    shap_vals_adv = np.array([pad_seq(x) for x in shap_vals_adv.values]).reshape(-1, 512)


    np.save('data/SHAP_signatures/shap_org.npy', shap_vals_org)
    np.save('data/SHAP_signatures/shap_adv.npy', shap_vals_adv)



    if save_path:
        pass


def parse_csv(csv_path):
    project_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(os.path.join(project_dir, csv_path))

    df['original_text'] = df['original_text'].apply(lambda x: re.sub('[\[\]]', '', x))
    df['perturbed_text'] = df['perturbed_text'].apply(lambda x: re.sub('[\[\]]', '', x))

    # Only return texts corresponding to successfull attacks
    df = df.loc[df['result_type'] == 'Successful']
    return df



if __name__ == '__main__':
    #samples = parse_csv('data/adversarial_samples/lstm_pwws_agnews.csv')
    #samples = parse_csv('data/adversarial_samples/distilbert_pwws_agnews.csv')
    #create_SHAP_signatures('lstm-ag-news', samples, 'ag_news')
    #create_SHAP_signatures('textattack/distilbert-base-uncased-ag-news', samples, 'ag_news')


    sample_paths = glob.glob("data/adversarial_samples/*.csv")

    for csv_path in sample_paths:
        csv = parse_csv(csv_path)

        try:
            # Extract the dataset name from the filename
            dset_name = re.search(r'_([a-zA-Z0-9-]*?).csv', csv_path).group(1)
        except AttributeError:
            raise Exception('Each adversarial csv file should match this naming convention: '
                            'architecture_attack_dataset.csv')


        samples = parse_csv(csv_path)


        if 'distilbert' in csv_path:
            model_name = 'textattack/distilbert-base-uncased-' + dset_name
        elif 'lstm' in csv_path:
            model_name = 'lstm-' + dset_name
        else:
            raise NotImplementedError('Architecture not implemented yet')

        if dset_name == 'ag-news':
            dset_name = 'ag_news'           # Required by datasets package

        create_SHAP_signatures(model_name, samples, dset_name)


