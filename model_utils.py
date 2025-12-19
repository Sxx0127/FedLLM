from task import Task
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForCausalLM, LlamaForQuestionAnswering, AutoModel
from torch import nn
import torch

class OneDCNNEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneDCNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.ReLU()
            # nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            # nn.ReLU(True),
            # nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=padding),
            # nn.ReLU(True),
            # nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=kernel_size, padding=padding),
            # nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)


class OneDCNNDecoder(nn.Module):
    def __init__(self, output_size, input_size):
        super(OneDCNNDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()
            # nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=kernel_size,
            #                    stride=stride, padding=padding, output_padding=output_padding),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=kernel_size,
            #                    stride=stride, padding=padding, output_padding=output_padding),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(hidden_channels, output_channels, kernel_size=kernel_size,
            #                    stride=stride, padding=padding, output_padding=output_padding),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class OneDCNNAutoencoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneDCNNAutoencoder, self).__init__()
        self.encoder = OneDCNNEncoder(input_size, output_size)
        self.decoder = OneDCNNDecoder(output_size, input_size)

    def forward(self, x):
        # encoded = self.encoder(x)
        # origin_shape = encoded.shape
        # x = nn.Flatten()(encoded)
        # input_size = x.shape[-1]
        # print(input_size)
        # exit(0)
        # x = nn.Linear(input_size, self.output_size)(x)
        # x = nn.Linear(self.output_size, input_size)(x)
        # x = x.reshape(origin_shape)
        # decoded = self.decoder(x)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_model_path(model: str):
    if model == 'roberta-large':
        model_path = './model/models--roberta-large/'
    elif model == 'roberta-base':
        model_path = './model/models--roberta-base/'
    elif model == 'distilbert-base-multilingual-cased':
        model_path = './model/models--distilbert-base-multilingual-cased/'
    elif model == 'distilbert-base-uncased':
        model_path = './model/models--distilbert-base-uncased/'
    elif model == 'distilbert-base-cased':
        model_path = './model/models--distilbert-base-cased/'
    elif model == 'llama-2-7B':
        model_path = './model/models--llama-2-7b-hf/'
    elif model == 'llama-3.2-1B':
        model_path = './model/models--llama-3.2-1b/'
    else:
        model_path = './model/models--llama-3-8b/'
    return model_path


def get_model_tokenizer(model: str, max_length: int):
    model_path = get_model_path(model=model)
    if 'llama' in model:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length, padding=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif model == 'roberta-large':
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length)
    return tokenizer


def get_model_lora(model: str, lora_alpha: int, lora_rank: int, num_labels: list, task: Task):
    model_path = get_model_path(model=model)
    model_pre, task_type = get_automodel(model_path=model_path, num_labels=num_labels, task=task)
    if model == 'distilbert-base-multilingual-cased' or model == 'distilbert-base-cased':
        target_modules = ["q_lin", "v_lin"]
    elif model == 'roberta-large' or model == 'roberta-base':
        target_modules = ["query", "value"]
    elif model == 'llama-2-7B':
        target_modules = ["q_proj", "v_proj"]
        model_pre.config.pad_token_id = 2
    else:
        target_modules = ["q_proj", "v_proj"]
        model_pre.config.pad_token_id = 128001

    untrain_part = get_untrain_part(model, task)
    peft_config = LoraConfig(
        r=lora_rank,
        target_modules=target_modules,
        task_type=task_type,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # init_lora_weights=False
        # use_rslora=True,
        # bias='lora_only'
    )
    peft_model = get_peft_model(model_pre, peft_config)
    trainable_parameters, _ = peft_model.get_nb_trainable_parameters()
    peft_model.print_trainable_parameters()
    _, param_num = set_requires_grad(untrain_part=untrain_part, model=peft_model)
    return peft_model, trainable_parameters, untrain_part


def get_automodel(model_path: str, num_labels: list, task: Task):
    if task == Task.SequenceClassification:
        model_pre = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels[0])
        task_type = TaskType.CAUSAL_LM
    elif task == Task.CausalLM:
        # model_pre = AutoModelForSeq2SeqLM.from_pretrained(model_path, is_decoder=True)
        model_pre = AutoModelForCausalLM.from_pretrained(model_path, is_decoder=True)
        task_type = TaskType.CAUSAL_LM
    elif task == Task.TokenClassification:
        if 'llama' in model_path:
            exit("Llama do not support task TokenClassification")
        model_pre = AutoModelForTokenClassification.from_pretrained(model_path, id2label=num_labels[0],
                                                                    label2id=num_labels[1])
        task_type = TaskType.CAUSAL_LM
    else:
        if 'llama' in model_path:
            model_pre = AutoModel.from_pretrained(model_path)
            model_pre.save_pretrained("./model/base_model")
            model_pre = AutoModelForQuestionAnswering.from_pretrained("./model/base_model")
            # model_pre = AutoModelForQuestionAnswering.from_pretrained(model_path)
        else:
            model_pre = AutoModelForQuestionAnswering.from_pretrained(model_path)
        task_type = TaskType.QUESTION_ANS
    return model_pre, task_type


def get_untrain_part(model: str, task: Task):
    if model == 'distilbert-base-multilingual-cased':
        task_train = {Task.SequenceClassification: ['classifier.bias', 'classifier.weight',
                                                    'pre_classifier.bias', 'pre_classifier.weight'],
                      Task.TokenClassification: ['classifier.bias', 'classifier.weight'],
                      Task.QuestionAnswering: ['qa_outputs.bias', 'qa_outputs.weight'],
                      Task.CausalLM: []}
    elif model == 'roberta-large' or model == 'roberta-base':
        task_train = {Task.SequenceClassification: ['classifier.dense.bias', 'classifier.dense.weight',
                                                    'classifier.out_proj.bias', 'classifier.out_proj.weight'],
                      Task.TokenClassification: ['classifier.bias', 'classifier.weight'],
                      Task.QuestionAnswering: ['qa_outputs.bias', 'qa_outputs.weight'],
                      Task.CausalLM: []}
    # elif 'llama' in model:
    else:
        task_train = {Task.SequenceClassification: ['score.weight'],
                      Task.QuestionAnswering: ['qa_outputs.bias', 'qa_outputs.weight']}
    return task_train[task]


def set_requires_grad(untrain_part: list, model):
    param_num = 0
    for name, param in model.named_parameters():
        if any(part in name for part in untrain_part):
            param.requires_grad = True
            param_num += param.numel()
        # for layer in untrain_part:
        #     if layer in name:
        #         param.requires_grad = True
        #         param_num += param.numel()
    return model, param_num
