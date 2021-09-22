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

from src.models.models import get_model_by_name
from src.models.models import evaluate_torch, evaluate_huggingface


def create_SHAP_signatures(model_name, samples, dset_name, save_path=None):
    model, tokenizer = get_model_by_name(model_name, with_wrapper=False)

    org_text = samples['original_text']
    adv_text = samples['perturbed_text']


    if isinstance(model, textattack.models.helpers.LSTMForClassification):
        evaluate_torch(model, tokenizer, adv_text, samples['ground_truth_output'])

        org_text = torch.Tensor(tokenizer(samples['original_text'].to_list()))
        adv_text = torch.Tensor(tokenizer(samples['perturbed_text'].to_list()))
        dataset = textattack.datasets.HuggingFaceDataset(dset_name, split='test')

        data = np.array(tokenizer(load_dataset('sst2', 'default')['test']['text']))
        background = data[np.random.choice(data.shape[0], 500, replace=False)]
        background = torch.tensor(background, dtype=torch.int)
        #masker = shap.maskers.Independent(data=background)
        explainer = shap.DeepExplainer(model, tokenizer)
    else:
        device = 0 if torch.cuda.is_available() else -1         # -1 defaults to CPU
        # We need all output scores for the SHAP calculation
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

        explainer = shap.Explainer(pipe)
        evaluate_huggingface(model, tokenizer, adv_text, samples['ground_truth_output'])

    shap_vals_org = explainer(org_text)
    shap_vals_adv = explainer(adv_text)


    if save_path:
        pass


def parse_csv(csv_path):
    project_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(os.path.join(project_dir, csv_path))

    df['original_text'] = df['original_text'].apply(lambda x: re.sub('[\[\]]', '', x))
    df['perturbed_text'] = df['perturbed_text'].apply(lambda x: re.sub('[\[\]]', '', x))
    df = df.loc[df['result_type'] == 'Successful']
    return df



if __name__ == '__main__':
    samples = parse_csv('data/adversarial_samples/ag_news_pwws_distilbert.csv')
    create_SHAP_signatures('textattack/distilbert-base-uncased-ag-news', samples, 'ag_news')
    #create_SHAP_signatures('textattack/distilbert-base-uncased-SST-2', samples, 'sst2')