import sys
import sys
import math

import numpy as np
import pandas as pd
from typing import List, Optional, Union
# from data.sampler import SubsetSequentialSampler
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
# from transformers.adapters import
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoModel
# from transformers.adapters import CompacterConfig
from xglue import _LANGUAGES
from model_utils import OneDCNNAutoencoder
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.nn.utils import prune

# Global variable
nlp_dataset = None


def TrainAutoEncoder(args):
    dataA = torch.tensor([])
    dataB = torch.tensor([])
    file_path = 'gradient/' + args.model + '_'
    for i in range(1, 51, 1):
        file = file_path + str(i) + '.pt'
        name_grad = torch.load(file)
        for name in name_grad.keys():
            if 'lora_A' in name:
                if dataA.numel() == 0:
                    dataA = name_grad[name].clone().view(-1)
                else:
                    dataA = torch.vstack((dataA, name_grad[name].view(-1)))
            elif 'lora_B' in name:
                if dataB.numel() == 0:
                    dataB = name_grad[name].clone().view(-1)
                else:
                    dataB = torch.vstack((dataB, name_grad[name].view(-1)))
    print(dataA.shape)  # torch.Size([1200, 24576])
    print(dataB.shape)  # torch.Size([1200, 24576])
    input_sizeA = dataA.shape[-1]
    output_sizeA = int(args.proportion * input_sizeA * 5)
    meanA, stdA = torch.mean(dataA, dim=0), torch.std(dataA, dim=0)
    dataA = (dataA - meanA) / stdA
    input_sizeB = dataB.shape[-1]
    output_sizeB = int(args.proportion * input_sizeB * 5)
    meanB, stdB = torch.mean(dataB, dim=0), torch.std(dataB, dim=0)
    dataB = (dataB - meanB) / stdA
    print("the output_size of A and B are ", output_sizeA, output_sizeB)
    batch_size = 32
    train_loaderA = DataLoader(TensorDataset(dataA), batch_size=batch_size, shuffle=True)
    train_loaderB = DataLoader(TensorDataset(dataB), batch_size=batch_size, shuffle=True)
    modelA = OneDCNNAutoencoder(input_sizeA, output_sizeA)
    modelB = OneDCNNAutoencoder(input_sizeB, output_sizeB)
    modelA = TrainModel(modelA, train_loaderA)
    modelB = TrainModel(modelB, train_loaderB)
    
    return modelA, modelB, meanA, stdA, meanB, stdB

def TrainModel(model, train_loader):
    num_epochs = 20
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in train_loader:
            inputs = inputs[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = criterion(outputs, inputs)
            loss = torch.norm(inputs - outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
    return model

# 定义剪枝函数
# robert-base(0.05) robert-large(0.02)
def prune_lora_layers(model, untrain_part, keep_ratio=0.02):
    mask = torch.tensor([])
    name_mask = {}
    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and ('lora' in name or any(part.split('.')[0] in name for part in untrain_part)):
        if isinstance(module, nn.Linear) and 'lora' in name:
            m = prune.random_unstructured(module, name='weight', amount=1 - keep_ratio)
            # m = prune.random_structured(module, name='weight', amount=1 - keep_ratio, dim=0)
            mask = torch.cat((mask, m.weight_mask.view(-1)))
            name_mask[name] = m.weight_mask
            # 移除剪枝参数
            prune.remove(module, 'weight')
    return mask, name_mask

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (int(self.indices[i]) for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Logger(object):
    def __init__(self, location):
        self.terminal = sys.stdout
        self.log = open(location, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def set_random_seed(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_classes(dataset: str) -> int:
    """
    Return the number of classes in the given dataset
    Args:
        dataset: Dataset name (e.g., cifar10, cifar100)
    Returns:
        the number of classes
    """
    n_classes = 0
    if dataset == '20_newsgroups':
        n_classes = 20
    elif dataset == 'xglue':
        n_classes = 10
    if not n_classes:
        print('No {} dataset in data directory'.format(dataset))
        exit(0)
    return n_classes


def partition_xglue(subdataset: str, model: str, n_parties: int, partition: bool,
                    alpha: float, max_length: int = 256):
    global nlp_dataset
    if model == 'roberta-base':
        model_path = './model/models--roberta-base/'
    elif model == 'distilbert-base-multilingual-cased':
        model_path = './model/models--distilbert-base-multilingual-cased/'
    elif model == 'llama-2-7B':
        model_path = './model/models--llama-2-7b/'
    else:
        model_path = './model/models--llama-3-8b/'

    raw_dataset = load_dataset('xglue.py', name=subdataset, cache_dir='./data/xglue')
    langs = _LANGUAGES['nc']
    multi_valid_sets = pd.DataFrame()
    multi_test_sets = pd.DataFrame()
    train = raw_dataset['train']
    train = pd.DataFrame(train)
    train['label'] = train['news_category']
    train['text'] = train['news_title'] + ' ' + train['news_body']
    train = Dataset.from_pandas(train).remove_columns(['news_title', 'news_body', 'news_category'])
    train = pd.DataFrame(train)

    for lange in tqdm(langs):
        vaild = raw_dataset['validation.{}'.format(lange)]
        vaild = pd.DataFrame(vaild)
        vaild['label'] = vaild['news_category']
        vaild['text'] = vaild['news_title'] + ' ' + vaild['news_body']
        vaild = Dataset.from_pandas(vaild).remove_columns(['news_title', 'news_body', 'news_category'])
        vaild = pd.DataFrame(vaild)
        multi_valid_sets = pd.concat([vaild, multi_valid_sets], axis=0)

        test = raw_dataset['test.{}'.format(lange)]
        test = pd.DataFrame(test)
        test['label'] = test['news_category']
        test['text'] = test['news_title'] + ' ' + test['news_body']
        test = Dataset.from_pandas(test).remove_columns(['news_title', 'news_body', 'news_category'])
        test = pd.DataFrame(test)
        multi_test_sets = pd.concat([test, multi_test_sets], axis=0)

    train_dataset = Dataset.from_pandas(train)
    vaild_dataset = Dataset.from_pandas(multi_valid_sets)
    test_dataset = Dataset.from_pandas(multi_test_sets)
    full_dataset = DatasetDict({'train': train_dataset, 'vaild': vaild_dataset, 'test': test_dataset})

    # Tokenization
    if 'llama' in model_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, max_length=max_length, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)

    n_samples = len(tokenized_dataset['train'])

    if partition:
        samples_per_party = (np.repeat(n_samples, n_parties) / n_parties).astype(int)
        print(samples_per_party)
    else:
        # 使用Dirichlet分布生成每个客户端的数据比例
        proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
        #
        # 计算每个客户端的样本数目
        samples_per_party = (proportions * n_samples).astype(int)

        # 找出零样本的客户端并加1
        zero_indices = np.argwhere(samples_per_party == 0).flatten()
        samples_per_party[zero_indices] += 1

        # 修正最后一个客户端的样本数以保证总数目匹配
        samples_per_party[-1] = n_samples - samples_per_party[:-1].sum()

        if samples_per_party[-1] <= 0:
            k = samples_per_party[-1]
            samples_per_party[-1] = 1
            max_element = np.max(samples_per_party)
            max_index = np.argmax(samples_per_party)
            max_element = max_element + k - 1
            samples_per_party[max_index] = max_element

    # 确保每个客户端至少有一个样本
    assert all(samples_per_party > 0), "Some parties ended up with no data. Consider changing the 'alpha' parameter."

    # 从训练集中获取所有样本的索引
    all_train_indices = np.arange(n_samples)

    # 随机打乱索引
    np.random.shuffle(all_train_indices)

    # 根据每个客户端的样本数目划分索引
    indices_split = np.split(all_train_indices, np.cumsum(samples_per_party)[:-1])

    # 创建每个客户端的数据映射（客户端ID -> 数据索引）
    net_dataidx_map = {i: indices_split[i].tolist() for i in range(n_parties)}

    nlp_dataset = tokenized_dataset.remove_columns(['text'])
    nlp_dataset.set_format("torch")
    for i, indices in enumerate(indices_split):
        client_dataset = full_dataset['train'].select(indices)
        net_dataidx_map[i] = indices.tolist()
        nlp_dataset[i] = {'train': client_dataset}
    return net_dataidx_map


def load_and_prepare_20newsgroups(dataset: str, datadir: str, model: str, partition: bool, n_parties: int = 10,
                                  alpha: float = 0.4, max_length: int = 256):
    global nlp_dataset
    # 加载数据集
    if model == 'roberta-base':
        model_path = './model/models--roberta-base/'
    elif model == 'distilbert-base-multilingual-cased':
        model_path = './model/models--distilbert-base-multilingual-cased/'
    elif model == 'llama-2-7B':
        model_path = './model/models--llama-2-7b/'
    else:
        model_path = './model/models--llama-3-8b/'
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # 转换为Pandas DataFrame
    train_df = pd.DataFrame({'text': newsgroups_train['data'], 'label': newsgroups_train['target']})
    test_df = pd.DataFrame({'text': newsgroups_test['data'], 'label': newsgroups_test['target']})

    # 转换为Hugging Face的Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    full_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    # Tokenization
    if 'llama' in model:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
    n_samples = len(tokenized_dataset['train'])
    if partition:
        samples_per_party = (np.repeat(n_samples, n_parties) / n_parties).astype(int)
        print(samples_per_party)
    else:
        # 使用Dirichlet分布生成每个客户端的数据比例
        proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
        #
        # 计算每个客户端的样本数目
        samples_per_party = (proportions * n_samples).astype(int)

        # 找出零样本的客户端并加1
        zero_indices = np.argwhere(samples_per_party == 0).flatten()
        samples_per_party[zero_indices] += 1

        # 修正最后一个客户端的样本数以保证总数目匹配
        samples_per_party[-1] = n_samples - samples_per_party[:-1].sum()

        if samples_per_party[-1] <= 0:
            k = samples_per_party[-1]
            samples_per_party[-1] = 1
            max_element = np.max(samples_per_party)
            max_index = np.argmax(samples_per_party)
            max_element = max_element + k - 1
            samples_per_party[max_index] = max_element

        # 确保每个客户端至少有一个样本
        assert all(
            samples_per_party > 0), "Some parties ended up with no data. Consider changing the 'alpha' parameter."

    # 从训练集中获取所有样本的索引
    all_train_indices = np.arange(n_samples)

    # 随机打乱索引
    np.random.shuffle(all_train_indices)

    # 根据每个客户端的样本数目划分索引
    indices_split = np.split(all_train_indices, np.cumsum(samples_per_party)[:-1])

    # 创建每个客户端的数据映射（客户端ID -> 数据索引）
    net_dataidx_map = {i: indices_split[i].tolist() for i in range(n_parties)}

    nlp_dataset = tokenized_dataset.remove_columns(['text'])
    nlp_dataset.set_format("torch")
    for i, indices in enumerate(indices_split):
        client_dataset = full_dataset['train'].select(indices)
        net_dataidx_map[i] = indices.tolist()
        nlp_dataset[i] = {'train': client_dataset}
    return net_dataidx_map, None


def load_and_prepare_20newsgroups_non_iid_label(model: str, n_parties: int = 10, alpha: float = 0.4,
                                                max_length: int = 256):
    global nlp_dataset
    # 加载数据集
    if model == 'roberta-base':
        model_path = '/home/wangjinzhi/fed_llm/fed_llm/model/models--roberta-base/'
    elif model == 'distilbert-base-multilingual-cased':
        model_path = './model/models--distilbert-base-multilingual-cased/'
    else:
        model_path = './model/models--llama-2-7b/'
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # 转换为Pandas DataFrame
    train_df = pd.DataFrame({'text': newsgroups_train['data'], 'label': newsgroups_train['target']})
    test_df = pd.DataFrame({'text': newsgroups_test['data'], 'label': newsgroups_test['target']})

    # 转换为Hugging Face的Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    full_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    # Tokenization
    if 'llama' in model:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, max_length=max_length, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
    n_samples = len(tokenized_dataset['train'])

    labels = np.array(newsgroups_train.target)
    n_classes = labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_classes, n_parties)

    proportions = np.array([1 / n_parties] * n_parties)
    samples_per_party = (proportions * n_samples).astype(int)
    samples_per_party[-1] = n_samples - samples_per_party[:-1].sum()
    copy_samples_per_party = samples_per_party
    indices_split = [[] for _ in range(n_parties)]
    label_distribution_cumsum = np.cumsum(label_distribution, axis=1)
    label_data = []
    for label in range(n_classes):
        indices = np.where(newsgroups_train.target == label)[0]
        label_data.append(indices)
    cls_amount = [len(label_data[i]) for i in list(range(len(label_data)))]
    while np.sum(samples_per_party) != 0:
        curr_clnt = np.random.randint(n_parties)
        if copy_samples_per_party[curr_clnt] <= 0:
            continue
        copy_samples_per_party[curr_clnt] -= 1
        curr_prior = label_distribution_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if trn_y is out of that class
            if cls_amount[cls_label] <= 0:
                continue
            cls_amount[cls_label] -= 1
            indices_split[curr_clnt].append(label_data[cls_label][cls_amount[cls_label]])
            break

    # 创建每个客户端的数据映射（客户端ID -> 数据索引）
    net_dataidx_map = {i: indices_split[i] for i in range(n_parties)}

    nlp_dataset = tokenized_dataset.remove_columns(['text'])
    nlp_dataset.set_format("torch")
    for i, indices in enumerate(indices_split):
        client_dataset = full_dataset['train'].select(indices)
        net_dataidx_map[i] = indices
        nlp_dataset[i] = {'train': client_dataset}
    return net_dataidx_map, None


def get_dateset():
    global nlp_dataset
    return nlp_dataset


def get_dataloader(dataset: str, datadir: str, train_bs: int, test_bs: int, dataidxs=None):
    train_dataset = nlp_dataset["train"]
    test_dataset = nlp_dataset["test"]
    if dataidxs is None:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
    else:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs),
                              pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
    return train_dl, test_dl


def get_multi_dataloader(dataset: str, datadir: str, train_bs: int, test_bs: int, dataidxs=None, client_id: int = None,
                         lang=None, n_parties=100):
    langs = ['en', 'de', 'es', 'fr', 'ru']

    if client_id is not None:
        n_clients_per_language = int(n_parties / len(langs))
        if n_clients_per_language == 0:
            n_clients_per_language = 1

        lang_idx = int(client_id / n_clients_per_language)
        current_lang = langs[lang_idx]
        train_dataset, test_dataset = nlp_dataset[current_lang]

    if lang is not None:
        train_dataset, test_dataset = nlp_dataset[lang]

    if n_parties == 1:  # Union mode
        all_lang_train_dataset = []
        all_lang_test_dataset = []
        for lang in langs:
            train_dataset, test_dataset = nlp_dataset[lang]
            all_lang_train_dataset.append(train_dataset)
            all_lang_test_dataset.append(test_dataset)
        all_lang_train_dataset = concatenate_datasets(all_lang_train_dataset)
        all_lang_test_dataset = concatenate_datasets(all_lang_test_dataset)

        train_dl = DataLoader(all_lang_train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(all_lang_test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(all_lang_train_dataset, batch_size=train_bs, pin_memory=True)

        return train_dl, test_dl, local_dl
    else:
        if dataidxs is None:
            train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
            test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
            local_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True)
        else:
            train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs),
                                  pin_memory=True)
            test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
            local_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetSequentialSampler(dataidxs),
                                  pin_memory=True)
    return train_dl, test_dl, local_dl


from peft import get_peft_model, TaskType


def initialize_networks(dataset: str, model: str, device: str = 'cpu', adapter: str = '', rank=16, con_dim=32,
                        task_type=TaskType.SEQ_CLS):
    model_path = model
    n_classes = get_num_classes(dataset)
    target_modules = ["query", "value"]
    adapter = adapter.lower()
    if model == 'distilbert-base-multilingual-cased':
        model_path = './model/models--distilbert-base-multilingual-cased/'
        target_modules = ["q_lin", "v_lin"]
        model_pre = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=n_classes)
    elif model == 'roberta-base':
        model_path = './model/models--roberta-base/'
        model_pre = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=n_classes)
        target_modules = ["query", "value"]
    elif model == 'llama-2-7B':
        model_pre = AutoModelForSequenceClassification.from_pretrained('./model/models--llama-2-7b/',
                                                                       num_labels=n_classes)
        target_modules = ["q_proj", "v_proj"]
        model_pre.config.pad_token_id = 2
    else:
        model_pre = AutoModelForSequenceClassification.from_pretrained('./model/models--llama-3-8b/',
                                                                       num_labels=n_classes)
        target_modules = ["q_proj", "v_proj"]
        model_pre.config.pad_token_id = 128001
    if adapter == 'lora':
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=rank,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=con_dim,
            lora_dropout=0.05
        )
        model = get_peft_model(model_pre, peft_config)
    elif adapter == 'prefix':
        from peft import PrefixTuningConfig
        peft_config = PrefixTuningConfig(task_type=task_type, )
        model = get_peft_model(model_pre, peft_config)
    elif adapter == 'prompt':
        from peft import PromptTuningConfig
        peft_config = PromptTuningConfig(task_type=task_type,
                                         )
        model = get_peft_model(model_pre, peft_config)
    elif adapter == 'p-tuning':
        from peft import PromptEncoderConfig
        peft_config = PromptEncoderConfig(task_type=task_type,
                                          )
        model = get_peft_model(model_pre, peft_config)
    else:
        model = model_pre

    model.print_trainable_parameters()
    return model
