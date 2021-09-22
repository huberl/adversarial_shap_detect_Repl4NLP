# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import transformers
import textattack
from textattack.attack_recipes import *
from textattack.models.helpers import LSTMForClassification

from src.models.models import get_model_by_name

ATTACKS = {
    'PWWS': 'PWWSRen2019',
    'BAE': 'BAEGarg2019'
}


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def create_adversarial_samples(model_name, dataset, num_samples, attack, split='test'):
    logger = logging.getLogger(__name__)
    logger.info(f'Creating {num_samples} for the {dataset} dataset on the {model_name} model using the {split} split')

    model, _ = get_model_by_name(model_name)

    csv_name = f'test'
    dataset = textattack.datasets.HuggingFaceDataset(dataset, split=split)
    attack = eval(ATTACKS[attack]).build(model)

    attack_args = textattack.AttackArgs(num_successful_examples=num_samples, log_to_csv='logger.csv', disable_stdout=True)

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    # DistilBERT + BAE
    create_adversarial_samples('textattack/distilbert-base-uncased-SST-2', dataset='gpt3mix/sst2', num_samples=2,
                               attack='BAE', split='test')
    '''create_adversarial_samples('textattack/distilbert-base-uncased-adversarial', dataset='ag_news', num_samples=1,
                               attack='BAE', split='test')
    create_adversarial_samples('textattack/distilbert-base-uncased-imdb', dataset='imdb', num_samples=1,
                               attack='BAE', split='test')

    # Bi-LSTM
    create_adversarial_samples('lstm-sst2', dataset='gpt3mix/sst2', num_samples=1, attack='BAE', split='test')
    create_adversarial_samples('lstm-adversarial', dataset='ag_news', num_samples=1, attack='BAE', split='test')
    create_adversarial_samples('lstm-imdb', dataset='imdb', num_samples=1, attack='BAE', split='test')'''


