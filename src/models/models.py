import textattack
import torch
import transformers
from textattack.models.helpers import LSTMForClassification
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline
from tqdm import tqdm

def evaluate_huggingface(model, tokenizer, inputs, labels):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    out = []
    # Transformers pipelines do not support batching at the moment
    for x in tqdm(inputs.to_list()):
        out.extend(pipe(x))

    out = np.array(out)
    return ([int(x['label'][-1]) for x in out] == labels).sum() / len(labels)


def evaluate_torch(model, tokenizer, inputs, labels):
    model.eval()

    inputs = tokenizer(inputs.to_list())
    train = torch.tensor(inputs)
    train_target = torch.tensor(labels.values.astype(np.float32))
    input_tensor = TensorDataset(train, train_target)
    loader = DataLoader(dataset=input_tensor, batch_size=128, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0

    model.to(device)


    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        out = model(inputs)
        pred = torch.softmax(out, dim=1).argmax(dim=1)
        correct += (pred == labels).sum()

    return correct / len(loader.dataset)


def get_model_by_name(model_name, with_wrapper=True):
    # Pretrained model from textattack
    if model_name.startswith('lstm'):
        model = LSTMForClassification.from_pretrained(model_name)
        tokenizer = model.tokenizer

        if with_wrapper:
            model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
    else:  # From transformers library
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        if with_wrapper:
            model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    return model, tokenizer