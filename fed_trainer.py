import os
import sys
import copy
import torch
import numpy as np
import math
import time
from model_utils import get_model_path, get_automodel, get_untrain_part, set_requires_grad
from torch import nn
from torch.nn.utils import prune

from task import Task
from fedcomp import Fedcomp
from accelerate import Accelerator
from torch.utils.data import Subset
from safetensors.torch import save_file
from cvlc import Fed_cvlc, get_coefficient_of_variation
from model_utils import get_model_tokenizer, get_model_lora
from dataset_utils import get_fed_data_info, Metrics_trainer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, DataCollatorForTokenClassification, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from utils import TrainAutoEncoder, prune_lora_layers


class Fed_trainer(object):
    def __init__(self, args, random_list):
        super().__init__()
        self.accelerator = Accelerator()
        self.data_collator = None
        self.metric_trainer = None
        self.gobal_model = None
        self.gobal_grad = None
        self.args = args
        self.training_metric = {}
        self.random_list = random_list
        self.accu_gra = None
        self.communication = 0

    def get_grad(self, model):
        grad = torch.tensor([])
        param = torch.tensor([])
        name_grad = {}
        name_param = {}
        name_paramlast = {}
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            # if 'lora' in layer or any(part in layer for part in self.untrain_part):
            if 'lora' in layer:
                if self.args.method == 'prune':
                    param_now = model.state_dict()[layer+"_orig"].detach().cpu()
                else:
                    param_now = model.state_dict()[layer].detach().cpu()
                param_last = self.gobal_model.state_dict()[layer].detach().cpu()
                param_g = param_last - param_now
                param = torch.cat((param, param_now.view(-1)))
                grad = torch.cat((grad, param_g.view(-1)))
                name_grad[layer] = param_g
                name_param[layer] = param_now
                name_paramlast[layer] = param_last
        # if self.args.method == 'raw':
        #     file_path = 'gradient/' + self.args.model + '_' + str(self.communication) + '.pt'
        #     torch.save(name_grad, file_path)
        return grad, param, name_grad, name_param, name_paramlast

    def paramAggre(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        grad = grad.cuda()
        current_index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if 'lora' in name:
                param = param.cuda()
                numel = param.data.numel()
                size = param.data.size()
                current_state_dict[name] = grad[current_index:current_index + numel].view(size)
                current_index += numel
        model.load_state_dict(current_state_dict)
        return model
    
    def substitute(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        grad = grad.cuda()
        current_index = 0
        index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if 'lora' in name:
                param = param.cuda()
                if index % 2 == 0:
                    _, mA = param.shape
                    nameA = name
                else:
                    mB, r = param.shape
                    current_grad = grad[current_index: current_index + mA * mB].reshape(mB, mA)
                    current_index += mA * mB
                    print("the shape of matrix is ", current_grad.shape)
                    print("the size and non-zero element of matrix are ", mA * mB, torch.sum(current_grad != 0))
                    
                    U, S, Vh = torch.svd_lowrank(current_grad.to(torch.float32), q=r)
                    print("the shape of U S V are ", U.shape, S.shape, Vh.shape)
                    S = torch.diag(S)
                    B = torch.matmul(U, S)
                    A = Vh.T
                    # B = torch.matmul(U, torch.sqrt(S))
                    # A = torch.matmul(torch.sqrt(S), Vh.T)
                    # U, S, Vh = torch.linalg.svd(current_grad.reshape(mB, mA))
                    # U_r = U[:, :r]
                    # S_r = torch.diag(S[:r])
                    # Vh_r = Vh[:r, :]
                    # B = torch.matmul(U_r, torch.sqrt(S_r))
                    # A = torch.matmul(torch.sqrt(S_r), Vh_r)

                    current_state_dict[nameA] = A
                    current_state_dict[name] = B
                index += 1
                # numel = param.data.numel()
                # size = param.data.size()
                # current_state_dict[name] = \
                #     torch.subtract(param.data.detach(), grad[current_index:current_index + numel].view(size))
                # current_index += numel
        model.load_state_dict(current_state_dict)
        return model
    
    def updateW(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        current_index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if any(target in name for target in self.target_modules) and 'weight' in name and 'lora' not in name:
                m, n = current_state_dict[name].shape
                print("the shape of {} is {}".format(name, (m, n)))
                lora_res = grad[current_index: current_index + m * n].detach()
                current_state_dict[name] += self.args.factor * self.args.lora_alpha / math.sqrt(self.args.lora_rank) * lora_res.reshape(m, n)
                current_index += m * n
            # elif any(part in name for part in self.untrain_part):
            #     numel = current_state_dict[name].numel()
            #     current_state_dict[name] -= grad[current_index: current_index + numel].reshape(current_state_dict[name].shape)
            #     current_index += numel
        # for name, param in current_state_dict.items():
        #     if 'lora_A' in name:
        #         # torch.nn.init.kaiming_normal_(current_state_dict[name])
        #         torch.nn.init.kaiming_uniform_(current_state_dict[name], a=math.sqrt(5 / 5)) # distilbert
        #         # torch.nn.init.normal_(current_state_dict[name], mean=0, std=1. / self.args.lora_rank)
        #     if 'lora_B' in name:
        #         torch.nn.init.zeros_(current_state_dict[name])
        model.load_state_dict(current_state_dict)
        return model
    
    def combine(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.gobal_model
        grad = grad.cuda()
        current_index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            # if 'lora' in name or any(part in name for part in self.untrain_part):
            if 'lora' in name:
                param = param.cuda()
                numel = param.data.numel()
                size = param.data.size()
                current_state_dict[name] = torch.subtract(param.data.detach(), 
                                                          grad[current_index:current_index + numel].view(size))
                current_index += numel
        model.load_state_dict(current_state_dict)
        return model

    def aggregate(self, grad_dist: dict, cohorts: list, partition_map: dict):
        model_gra = torch.zeros_like(grad_dist[cohorts[0]])
        data_sum = 0
        for client in cohorts:
            data_sum += len(partition_map[client])
        for client in cohorts:
            w = len(partition_map[client]) / data_sum
            model_gra += (w * grad_dist[client])
        return model_gra

    def set_data_collator(self, tokenizer, task):
        if task == Task.SequenceClassification:
            self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        elif task == Task.TokenClassification:
            self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        elif task == Task.QuestionAnswering:
            self.data_collator = None
        else:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def run(self):
        print("configaration is ", self.args)
        tokenizer = get_model_tokenizer(model=self.args.model, max_length=self.args.max_length)
        data, partition_map, num_labels, metric, validation_key, \
            task, label_names, validation_dataset, grids = get_fed_data_info(args=self.args, tokenizer=tokenizer)
        self.metric_trainer = Metrics_trainer(metric_name=metric, label_names=label_names, grids=grids)

        if self.args.method == 'updateW':
            if self.args.model == 'distilbert-base-multilingual-cased' or self.args.model == 'distilbert-base-cased':
                self.target_modules = ["q_lin", "v_lin"]
            elif self.args.model == 'roberta-large' or self.args.model == 'roberta-base':
                self.target_modules = ["query", "value"]
            elif self.args.model == 'llama-2-7B' or self.args.model == 'llama-3.2-1B':
                self.target_modules = ["q_proj", "v_proj"]
            else:
                self.target_modules = ["q_proj", "v_proj"]
        self.gobal_model, trainable_parameters, self.untrain_part = get_model_lora(model=self.args.model, lora_alpha=self.args.lora_alpha,
                                                                lora_rank=self.args.lora_rank, num_labels=num_labels,
                                                                task=task)
        print("the model architecture is ")
        print(self.gobal_model)

        num_param = 0
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer:
                num_param += self.gobal_model.state_dict()[layer].numel()
        self.accu_gra = torch.zeros((self.args.num_client, num_param))
        self.set_data_collator(tokenizer=tokenizer, task=task)
        matrix = self.get_model_matrix_num()
        traffic = int(self.args.proportion * trainable_parameters) * (32 + math.log2(trainable_parameters) + 1)
        self.args.packet_num = int(traffic / 1500 / 8)
        # self.args.packet_num = self.get_packet_num(layers=matrix)
        print("the number of packet is ", self.args.packet_num)
        
        # coefficient_of_variation = []
        self.init_training_metric(metrics=metric)
        compress = Fed_cvlc(args=self.args, trainable_parameters=trainable_parameters, matrix_num=matrix)
        residual_model = {}
        if self.args.method == 'prune':
            prune_propor = int(traffic / 32) / trainable_parameters
            self.mask, self.name_mask = prune_lora_layers(self.gobal_model, self.untrain_part, keep_ratio=prune_propor)
            print("the proportion of mask and prune_propor are ", 
                  torch.count_nonzero(self.mask) / self.mask.numel(), prune_propor)

        # Train
        for rnd in range(self.args.comm_round):
            if self.accelerator.is_local_main_process:
                print(f'ROUND:{rnd}')
            np.random.seed(self.random_list[rnd])
            cohorts = np.random.choice(self.args.num_client, int(self.args.num_client * self.args.sample_fraction),
                                    replace=False).tolist()
            grad_dist = {}
            for i, client in enumerate(cohorts):
                self.accelerator.print(f'CLIENT:{client}')
                local_model = copy.deepcopy(self.gobal_model)
                if self.args.residual:
                    if client in residual_model and residual_model[client] is not None:
                        local_model = self.combine(residual_model[client], copy.deepcopy(local_model))
                local_model = self.train(data=data, data_indices=partition_map[client],
                                        model=copy.deepcopy(local_model),
                                        tokenizer=tokenizer, task=task, round=rnd)
                if rnd == 0 and client == cohorts[0]:
                    self.copy_model_expect_lora(local_model)
                grad, param, name_grad, name_param, name_paramlast = self.get_grad(local_model)
                if self.args.method in ['fedcomp', 'smartidx']:
                    fedcomp = Fedcomp(args=self.args)
                    grad_dist[client], res_model = fedcomp.fed_comp(self.gobal_model, local_model)
                    if self.args.residual:
                        residual_model[client] = res_model
                # elif self.args.method == 'new':
                #     grad_dist[client] = compress.new(name_param, proportion=self.args.proportion)
                elif self.args.method == 'updateW':
                    if rnd == 0 and i == 0:
                        opt = True
                    else:
                        opt = False
                    grad_dist[client] = compress.STopK(param, grad, name_param, name_grad, proportion=self.args.proportion, opt=opt, rnd=rnd)
                # elif self.args.method == 'CGFedLLM':
                #     grad_dist[client] = compress.CGFedLLM(name_grad, autoencoderA, autoencoderB, meanA, stdA, meanB, stdB)
                elif self.args.method == 'compeft':
                    grad += self.accu_gra[client]
                    grad_dist[client] = compress.compeft(self.args.proportion, name_grad, grad)
                    self.accu_gra[client] = grad - grad_dist[client]
                elif self.args.method == 'prune':
                    # mask = prune_lora_layers(self.gobal_model, self.untrain_part, keep_ratio=self.args.proportion * 2)
                    # grad_dist[client] = compress.prune(grad, self.mask)
                    grad_dist[client] = grad
                    print("the proportion of non-zeros is ", torch.count_nonzero(grad) / grad.numel())
                    # if rnd <= 1:
                    #     grad_dist[client] = compress.topk_AB(self.args.proportion, name_grad, grad)
                    # else:
                    #     mask = prune_lora_layers(self.gobal_model, self.untrain_part, keep_ratio=self.args.proportion * 2)
                    #     grad_dist[client] = compress.prune(grad, mask)
                else:
                    grad += self.accu_gra[client]
                    grad_dist[client] = compress.do_compress(grad, param, name_grad, name_paramlast, proportion=self.args.proportion, rnd=rnd)
                    self.accu_gra[client] = grad - grad_dist[client]
            grad = self.aggregate(grad_dist=grad_dist, cohorts=cohorts, partition_map=partition_map)
            if self.args.method == 'updateW':
                self.gobal_grad = grad
                if (rnd + 1) < self.args.point:
                # if (rnd + 1) % self.args.point != 0:
                    self.gobal_model = self.updateW(grad)
                else:
                    self.gobal_model = self.combine(grad)
                
                # gobal_model = self.updateW(grad)
                # self.gobal_model, trainable_parameters = get_model_lora(model=self.args.model, lora_alpha=self.args.lora_alpha,
                #                                                 lora_rank=self.args.lora_rank, num_labels=num_labels,
                #                                                 task=task)
                # state_dict = self.gobal_model.state_dict()
                # for key, _ in state_dict.items():
                #     if key.find('lora') == -1:
                #         state_dict[key] = gobal_model[key]
                #         state_dict[key].requires_grad = False
                #         # state_dict[key] = torch.zeros_like(state_dict[key])
                # self.gobal_model.load_state_dict(state_dict)
            else:
                self.gobal_grad = grad
                self.gobal_model = self.combine(grad)
            result, predictions = self.test_metric(data=data, tokenizer=tokenizer,
                                                validation_key=validation_key, task=task)
            if predictions is not None:
                result = self.metric_trainer.compute_metrics_predictions(predictions=predictions, data=data,
                                                                        result=result,
                                                                        validation_dataset=validation_dataset)
            self.record_metric(metrics=metric, result=result)

            if self.args.early_stop:
                if "accuracy" in metric and result['eval_acc'] >= 0.6:
                    print(f"Stopping training as accuracy has reached {result['eval_acc']} after {rnd + 1} rounds.")
                    break
        # if self.accelerator.is_local_main_process:
        #     if self.args.save_model:
        #         self.save_model()
        #     if self.args.save_metric:
        #         self.save_metric(metrics=metric)
        # with open('./save/cov_72.txt', 'w') as file:
        #     # 遍历列表中的每个元素，并将其转换为字符串后写入文件
        #     for item in coefficient_of_variation:
        #         file.write(str(item) + '\n')
        return

    def train(self, data, data_indices, model, tokenizer, task, round):
        if self.args.method == 'prune':
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and 'lora' in name:
                    tmp_mask = self.name_mask[name]
                    prune.CustomFromMask.apply(module, 'weight', tmp_mask)
        
        model.train()
        train_data = Subset(data["train"], data_indices)
        save_steps = sys.maxsize
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        lr = self.args.lr
        weight = self.args.weight
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=weight, betas=(0.9, 0.999), amsgrad=True)
        if task in [Task.SequenceClassification, Task.TokenClassification, Task.QuestionAnswering, Task.CausalLM]:
            training_args = TrainingArguments(output_dir='./save/model', save_steps=save_steps,
                                            #   save_strategy='epoch',
                                              num_train_epochs=self.args.epochs,
                                              per_device_train_batch_size=self.args.batch_size, do_train=True,
                                              learning_rate=lr,
                                              ddp_find_unused_parameters=False,
                                              lr_scheduler_type="constant",
                                              logging_steps=1)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_data,
                data_collator=self.data_collator,
                optimizers=(optimizer, None)
            )
        else:
            training_args = Seq2SeqTrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                                     num_train_epochs=self.args.epochs,
                                                     per_device_train_batch_size=self.args.batch_size, do_train=True,
                                                     learning_rate=self.args.lr,
                                                     lr_scheduler_type="constant",
                                                     ddp_find_unused_parameters=False,
                                                     logging_steps=1)
            trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_data,
                data_collator=self.data_collator,
                optimizers=(optimizer, None)
            )
        trainer.train()
        model.cpu()
        return model

    def test_metric(self, data, tokenizer, validation_key, task):
        self.gobal_model.eval()
        predictions = None
        eval_data = data[validation_key]
        if task in [Task.SequenceClassification, Task.TokenClassification]:
            training_args = TrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                              per_device_eval_batch_size=self.args.batch_size,
                                              do_eval=True, seed=self.args.random_seed)
            trainer = Trainer(
                model=self.gobal_model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=self.metric_trainer.compute_metrics,
                eval_dataset=eval_data
            )
            results = trainer.evaluate(metric_key_prefix="eval")
        elif task in [Task.QuestionAnswering]:
            training_args = TrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                              per_device_eval_batch_size=self.args.batch_size,
                                              do_predict=True, seed=self.args.random_seed)
            trainer = Trainer(
                model=self.gobal_model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=self.metric_trainer.compute_metrics,
                eval_dataset=eval_data
            )
            predictions, _, results = trainer.predict(eval_data)
        else:
            training_args = Seq2SeqTrainingArguments(output_dir='save', save_steps=sys.maxsize,
                                                     per_device_eval_batch_size=self.args.batch_size,
                                                     save_total_limit=1,
                                                     do_predict=True, seed=self.args.random_seed)
            trainer = Seq2SeqTrainer(
                model=self.gobal_model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=self.metric_trainer.compute_metrics,
                eval_dataset=eval_data
            )
            results = trainer.evaluate(max_length=self.args.max_length + 3)
        if self.accelerator.is_local_main_process:
            print(results)
        self.gobal_model.cpu()
        return results, predictions

    def save_model(self):
        file_name = '{}_{}_{}_{}_{}.safetensors'.format(self.args.model, self.args.alpha, self.args.num_client,
                                                        self.args.epochs, self.args.comm_round)
        save_file(self.gobal_model.state_dict(), os.path.join('./save', file_name))

    def save_metric(self, metrics: list):
        for metric in metrics:
            if metric == "matthews_correlation":
                metric_list = self.training_metric['matthews_correlation']
                name = "matthews_correlation_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method,
                                                                              self.args.blocks,
                                                                              self.args.alpha, self.args.comm_round,
                                                                              self.args.bit,
                                                                              self.args.dataset, self.args.subdataset)
                save_file_metric(name, metric_list)
            elif metric == "pearson":
                metric_list = self.training_metric['pearson']
                name = "pearson_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                    self.args.alpha, self.args.comm_round,
                                                                    self.args.bit,
                                                                    self.args.dataset, self.args.subdataset,
                                                                    self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "spearman":
                metric_list = self.training_metric['spearman']
                name = "spearman_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                     self.args.alpha, self.args.comm_round,
                                                                     self.args.bit,
                                                                     self.args.dataset, self.args.subdataset,
                                                                     self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "accuracy":
                metric_list = self.training_metric['accuracy']
                name = "accuracy_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                     self.args.alpha, self.args.comm_round,
                                                                     self.args.bit,
                                                                     self.args.dataset, self.args.subdataset,
                                                                     self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "f1" or metric == "f1_m":
                metric_list = self.training_metric['f1']
                name = "f1_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                               self.args.alpha, self.args.comm_round, self.args.bit,
                                                               self.args.dataset, self.args.subdataset,
                                                               self.args.model)
                save_file_metric(name, metric_list)
            elif metric == "precision":
                metric_list = self.training_metric['precision']
                name = "precision.txt"
                save_file_metric(name, metric_list)
            elif metric == "exact_match":
                metric_list = self.training_metric['exact_match']
                name = "exact_match_{}_{}_{}_{}_{}_{}_{}.txt".format(self.args.method, self.args.blocks,
                                                                     self.args.alpha, self.args.comm_round,
                                                                     self.args.bit,
                                                                     self.args.dataset, self.args.subdataset)
                save_file_metric(name, metric_list)
            elif metric == "nDCG":
                metric_list = self.training_metric['ndcg']
                name = "ndcg.txt"
                save_file_metric(name, metric_list)
            else:
                exit('No metric {} erro in save_metric'.format(metric))

        return

    def init_training_metric(self, metrics: list):
        for metric in metrics:
            if metric == "f1_m":
                metric = 'f1'
            self.training_metric[metric] = []

    def record_metric(self, metrics: list, result: dict):
        for metric in metrics:
            if metric == "matthews_correlation":
                self.training_metric[metric].append(result['eval_matthews_correlation'])
            elif metric == "pearson":
                self.training_metric[metric].append(result['eval_pearson'])
            elif metric == "spearman":
                self.training_metric[metric].append(result['eval_spearmanr'])
            elif metric == "accuracy":
                self.training_metric[metric].append(result['eval_acc'])
            elif metric in ["f1", "f1_m"]:
                metric = 'f1'
                self.training_metric[metric].append(result['eval_f1'])
            elif metric == "precision":
                self.training_metric[metric].append(result['eval_precision'])
            elif metric == "exact_match":
                self.training_metric[metric].append(result['eval_exact_match'])
            elif metric == "nDCG":
                self.training_metric[metric].append(result['eval_ndcg'])
            else:
                exit('No metric {} erro in record_metric'.format(metric))

    def get_model_matrix_num(self):
        num = 0
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer:
                num += 1
        return num
    
    def get_packet_num(self, layers):
        traffic = 0
        for layer in self.gobal_model.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer or any(part in layer for part in self.untrain_part):
                num = self.gobal_model.state_dict()[layer].numel()
                traffic += int(self.args.proportion * num) * (32 + math.log2(num) + math.log2(layers))
        return int(traffic / 1500 / 8)

    def copy_model_expect_lora(self, model):
        state_dict_a = self.gobal_model.state_dict()
        state_dict_b = model.state_dict()

        for name, param in state_dict_b.items():
            if 'lora' not in name and name in state_dict_b:
                state_dict_a[name].copy_(param)
        self.gobal_model.load_state_dict(state_dict_a)
        for name, param in self.gobal_model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False


def save_file_metric(name: str, metric_list: list):
    path = os.path.join('./newsave', name)
    with open(path, 'w') as file:
        for item in metric_list:
            file.write(str(item) + '\n')


def custom_deepcopy(module):
    """自定义深拷贝函数，处理权重归一化和剪枝"""
    if isinstance(module, nn.Module):
        # 创建新模块实例
        new_module = type(module)(*[getattr(module, param) for param in module.__constants__])
        new_module.load_state_dict(module.state_dict())
        
        # 处理子模块
        for name, child in module.named_children():
            setattr(new_module, name, custom_deepcopy(child))
        
        # 处理缓冲区（如掩码）
        for buffer_name, buffer in module.named_buffers():
            new_module.register_buffer(buffer_name, buffer.clone())
        
        return new_module
    return copy.deepcopy(module)