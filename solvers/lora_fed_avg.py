import copy
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from tqdm import tqdm

from utils import get_dataloader

from solvers.full_fed_avg import FullFedAvgSolver
from safetensors.torch import save_file
from torch.cuda.amp import GradScaler
from transformers import Trainer, TrainingArguments


class HyperFedAvgSolver(FullFedAvgSolver):
    def __init__(self, args, client2data: dict, cls_dist=None):
        super().__init__(args, client2data, cls_dist)
        self.best_loss = None
        self.best_bs = None
        self.best_ks = None
        self.best_k = None
        self.k_max = None
        self.k_min = None
        self.d = None
        self.accu_gra = None
        self.packet_size = 1500 * 8
        # 需要按层还是按矩阵来分配这个包的个数
        self.packet_num = 72
        self.matrix = None
        self.beta_1 = 0.85
        self.beta_2 = 0.85
        self.T_previous = {}
        self.U_previous = {}
        self.gradient = True
        self.graproduct = False
        self.graproduct_2 = False
        self.adalora = False

    def set_accu_gra(self):
        total_trainable_params = 0
        m = 0
        for p in self.global_net.state_dict():
            # for p in self.model.parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            if 'lora' in p:
                total_trainable_params += self.global_net.state_dict()[p].numel()
                # 按矩阵
                m += 1

        self.accu_gra = torch.zeros((self.args.n_parties, total_trainable_params)).to(self.device)
        # self.accu_gra_layer = torch.zeros((self.args.n_parties, total_trainable_params)).to(self.device)
        # self.accu_gra_matrix = torch.zeros((self.args.n_parties, total_trainable_params)).to(self.device)

        self.d = total_trainable_params
        self.matrix = m
        s = int(math.log(self.d, 2)) + 1
        print(s)
        self.k_min = int(self.packet_size / (s + 18)) * self.packet_num
        self.k_max = int(self.packet_size / (s + 5)) * self.packet_num

        self.accu_gra.cpu()

    def testing_multi_lang_plm(self, model=None):
        writer = {'loss': 0., 'acc': 0., 'step': 0}
        if model is None:
            model = self.global_net
        net = model.to(self.device)
        net.eval()

        total_acc = 0.
        writer['loss'] = 0.
        writer['acc'] = 0.
        writer['step'] = 0.
        writer['avg_code'] = 0.
        _, global_loader = get_dataloader(dataset=self.args.dataset,
                                          datadir=self.args.datadir,
                                          train_bs=self.args.batch_size,
                                          test_bs=self.args.batch_size)

        with torch.no_grad():
            for batch in tqdm(global_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # indexes = batch['id'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = net(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Summary
                writer['acc'] += torch.eq(logits.argmax(dim=-1), labels).float().sum()
                writer['step'] += len(logits)
            total_acc += float(writer['acc'] / writer['step'])
        return total_acc

    def training_plm(self, client_id: int, networks, test_loader=None, warmup=True, class_tokens=None):
        """ Perform local optimization (Training global model on the local data """
        # Load the optimizer
        optimizer = torch.optim.AdamW(networks.parameters(), lr=self.args.lr)
        scaler = GradScaler()
        # print('* Training Client ({}/{}) on {} data'.format(client_id + 1, self.args.n_parties,
        # len(self.local_train_loader[client_id]) * self.args.batch_size)

        data_labels = []
        for batch in self.local_train_loader[client_id]:
            labels = batch['label'].to(self.device)
            data_labels.append(labels)
        data_labels = torch.cat(data_labels, dim=0)
        data_labels = F.one_hot(data_labels, self.num_class).float().to(self.device).sum(dim=0, keepdim=True)
        # data_labels = data_labels / data_labels.sum(dim=-1, keepdim=True)
        networks.to(self.device)
        for epoch in range(self.args.epochs):
            writer = {'loss': 0., 'acc': 0., 'step': 0}
            networks.train()
            for name, param in networks.named_parameters():
                if name.startswith('classifier') or name.startswith(
                        'pre_classifier') or self.args.adapter in name:  # choose whatever you like here
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for batch in tqdm(self.local_train_loader[client_id]):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = networks(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = (-F.one_hot(labels, self.num_class) * logits.log_softmax(dim=-1)).sum(dim=-1)
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                writer['loss'] += loss.mean().item()
                writer['acc'] += torch.eq(logits.argmax(dim=-1), labels).float().mean()
                writer['step'] += 1

            # print(writer['acc'] / self.args.epochs)
            # if 'llama' in self.args.model:
            #     print(
            #         'Client {} Epoch ({}/{})'.format(client_id, epoch + 1, self.args.epochs))
            # else:
            #     test_acc = self.testing_multi_lang_plm(model=networks)
            #     print(
            #  'Client {} Epoch ({}/{}) Test accuracy {}'.format(client_id, epoch + 1, self.args.epochs, test_acc))
        networks = networks.cpu()

    def PQ_loss(self, x, xb, parameter):
        return x / (2 * ((2 ** xb - 1) ** 2)) * torch.sum(parameter)

    def topk(self, network, k=50000):
        parameters = torch.tensor([]).to(self.device)
        net = copy.deepcopy(network)
        net.to(self.device)
        for p in net.state_dict():
            if p.find("num_batches_tracked") != -1:
                continue
            if 'lora' in p:
                param1 = net.state_dict()[p].detach().to(self.device)
                param2 = self.global_net.state_dict()[p].detach().to(self.device)
                param3 = param1 - param2
                parameters = torch.cat((parameters, param3.view(-1)))
        _, indices = torch.topk(parameters.abs(), k)
        mask = torch.zeros_like(parameters)
        for i in indices:
            mask[i.item()] = 1
        parameters = parameters * mask
        return parameters, mask

    # def topk_total_layer(self, network):
    #     # 按矩阵
    #     if self.matrix_top:
    #         k = self.k_max // self.matrix
    #     elif self.layer_top:
    #         k = self.k_max // self.layer
    #
    #     def apply_topk(tensor, k):
    #         # 展平tensor并应用top-k
    #         flat_tensor = tensor.view(-1)
    #         top_values, top_indices = torch.topk(flat_tensor.abs(), k=min(k, flat_tensor.numel()), largest=True)
    #         mask = torch.zeros_like(flat_tensor)
    #         mask[top_indices] = 1
    #         return flat_tensor * mask
    #
    #     net = copy.deepcopy(network)
    #     net.to(self.device)
    #     topk_parameters = torch.tensor([]).to(self.device)
    #
    #     for p in net.state_dict():
    #         if "num_batches_tracked" in p:
    #             continue
    #         if 'lora' in p:
    #             param1 = net.state_dict()[p].detach().to(self.device)
    #             param2 = self.global_net.state_dict()[p].detach().to(self.device)
    #             gradient = param1 - param2
    #
    #             if self.gradient:
    #                 preserved_param = apply_topk(gradient, k)
    #             elif self.graproduct:
    #                 grad_param_product = param1 * gradient
    #                 preserved_param = apply_topk(grad_param_product, k)
    #             elif self.graproduct_2:
    #                 grad_param_product_squared = (param1 * gradient).pow(2)
    #                 preserved_param = apply_topk(grad_param_product_squared, k)
    #             elif self.adalora:
    #                 grad_param_product = param1 * gradient
    #                 flat_product = grad_param_product.view(-1)
    #                 T_current = self.beta_1 * self.T_previous.get(p, torch.zeros_like(flat_product)) + (
    #                         1 - self.beta_1) * flat_product
    #                 U_current = self.beta_2 * self.U_previous.get(p, torch.zeros_like(flat_product)) + (
    #                         1 - self.beta_2) * (flat_product - T_current).abs()
    #                 self.T_previous[p] = T_current
    #                 self.U_previous[p] = U_current
    #                 importance_score = T_current * U_current
    #                 preserved_param = apply_topk(importance_score, k)
    #
    #             topk_parameters = torch.cat((topk_parameters, preserved_param))
    #
    #     return topk_parameters

    def topk_matrix(self, network, k):
        def apply_topk(tensor, k):
            # 展平tensor并应用top-k
            flat_tensor = tensor.view(-1)
            init_value, init_indices = torch.topk(flat_tensor.abs(), k=min(k, flat_tensor.numel()))
            # mask = torch.zeros_like(flat_tensor)
            # for i in init_indices:
            #     mask[i.item()] = 1
            # df = pd.DataFrame(mask.cpu())
            # df.to_csv('matrix22.csv', index=False, header=False)
            abs_gra = flat_tensor.abs()
            return init_value, init_indices, abs_gra, flat_tensor  # 直接返回topk的值和索引

        net = copy.deepcopy(network)
        net.to(self.device)
        topk_matrix = {}

        for p in net.state_dict():
            if "num_batches_tracked" in p or 'lora' not in p:
                continue
            param1 = net.state_dict()[p].detach().to(self.device)
            param2 = self.global_net.state_dict()[p].detach().to(self.device)
            gradient = param1 - param2

            if self.gradient:
                preserved_values, preserved_indices, abs_gra, flat_tensor = apply_topk(gradient, k)
            elif self.graproduct:
                grad_param_product = param1 * gradient
                preserved_values, preserved_indices, abs_gra, flat_tensor = apply_topk(grad_param_product, k)
            elif self.graproduct_2:
                grad_param_product_squared = (param1 * gradient).pow(2)
                preserved_values, preserved_indices, abs_gra, flat_tensor = apply_topk(grad_param_product_squared, k)
            elif self.adalora:
                grad_param_product = param1 * gradient
                flat_product = grad_param_product.view(-1)
                T_current = self.beta_1 * self.T_previous.get(p, torch.zeros_like(flat_product)) + (
                        1 - self.beta_1) * flat_product
                U_current = self.beta_2 * self.U_previous.get(p, torch.zeros_like(flat_product)) + (
                        1 - self.beta_2) * (flat_product - T_current).abs()
                self.T_previous[p] = T_current
                self.U_previous[p] = U_current
                importance_score = T_current * U_current
                preserved_values, preserved_indices, abs_gra, flat_tensor = apply_topk(importance_score, k)
            else:
                exit(0)

            topk_matrix[p] = (preserved_values, preserved_indices, abs_gra, flat_tensor)

        return topk_matrix

    def topk_layer(self, network, k):

        def apply_topk(tensor, k):
            flat_tensor = tensor.view(-1)
            init_value, init_indices = torch.topk(flat_tensor.abs(), k=min(k, flat_tensor.numel()), largest=True)
            abs_gra = flat_tensor.abs()
            # mask = torch.zeros_like(flat_tensor)
            # for i in init_indices:
            #     mask[i.item()] = 1
            # df = pd.DataFrame(mask.cpu())
            # df.to_csv('matrix33.csv', index=False, header=False)
            return init_value, init_indices, abs_gra, flat_tensor

        net = copy.deepcopy(network)
        net.to(self.device)
        topk_layers = {}

        # 将参数按四个一组分组
        layer_tensors = []
        current_tensors = []
        current_names = []  # 用于存储当前参数的名称
        count = 0

        for p, param in net.state_dict().items():
            if "num_batches_tracked" in p or 'lora' not in p:
                continue

            current_tensors.append(param)
            current_names.append(p)  # 存储名称
            count += 1

            # 每四个参数处理一次
            if count % 4 == 0:
                # 聚合这四个矩阵的梯度
                gradients = [x.detach() - self.global_net.state_dict()[name].detach().to(self.device) for x, name in
                             zip(current_tensors, current_names)]
                # 将所有梯度展平并拼接
                flat_tensor = torch.cat([g.view(-1) for g in gradients])
                # 应用topk到聚合后的梯度
                preserved_values, preserved_indices, abs_gra, flat_tensor = apply_topk(flat_tensor, k)
                layer_name = f"layer_{len(topk_layers) + 1}"
                topk_layers[layer_name] = (preserved_values, preserved_indices, abs_gra, flat_tensor)

                # 重置当前层的参数列表
                current_tensors = []
                current_names = []  # 清空名称列表

        return topk_layers

    def cvlc_layer(self, client_id, network, matrix_top=False, layer_top=False):
        k_sum = 0
        if matrix_top:
            positionID = self.d / self.matrix
            s = int(math.log(positionID, 2)) + 1
            kk_min = int(self.packet_size / (s + 18)) * self.packet_num
            kk_max = int(self.packet_size / (s + 5)) * self.packet_num
            # topk_results = self.topk_matrix(network, kk_max)

            k_max = kk_max // self.matrix
            k_min = kk_min // self.matrix
            topk_results = self.topk_matrix(network, k_max)

            new_grads_matrix = torch.zeros_like(self.accu_gra[client_id]).to(self.device)
            start_index = 0
            for name, (init_value, init_indices, abs_gra, flat_tensor) in topk_results.items():
                original_grad = flat_tensor
                param_size = original_grad.numel()

                optimized_grad_matrix, k_matrix = self.apply_cvlc(init_value, init_indices, original_grad,
                                                                  abs_gra, k_max, k_min, s, matrix_top, layer_top)
                k_sum += k_matrix

                new_grads_matrix[start_index:start_index + param_size] = optimized_grad_matrix

                start_index += param_size
            print(k_sum)
            return new_grads_matrix  # 如果combine方法中使用了torch.add，这里需要乘以-1

        if layer_top:
            layer = self.matrix // 4
            positionID = self.d / layer
            s = int(math.log(positionID, 2)) + 1
            kk_min = int(self.packet_size / (s + 18)) * self.packet_num
            kk_max = int(self.packet_size / (s + 5)) * self.packet_num

            # 均匀分配，每一层一样包的数目
            k_max = kk_max // layer
            k_min = kk_min // layer
            topk_results = self.topk_layer(network, k_max)

            new_grads_layer = torch.zeros_like(self.accu_gra[client_id]).to(self.device)

            start_index = 0
            for name, (init_value, init_indices, abs_gra, flat_tensor) in topk_results.items():
                original_grad = flat_tensor
                param_size = original_grad.numel()

                optimized_grad_layer, k_layer = self.apply_cvlc(init_value, init_indices, original_grad,
                                                                abs_gra, k_max, k_min, s, matrix_top, layer_top)
                k_sum += k_layer

                # 统计信息
                # mean_optimized_grad = torch.mean(optimized_grad).item()
                # std_optimized_grad = torch.std(optimized_grad).item()
                # min_optimized_grad = torch.min(optimized_grad).item()
                # max_optimized_grad = torch.max(optimized_grad).item()
                #
                # # 打印统计信息
                # print(f"Layer: {name}")
                # print(f"Mean of optimized grad: {mean_optimized_grad}")
                # print(f"Std of optimized grad: {std_optimized_grad}")
                # print(f"Min of optimized grad: {min_optimized_grad}")
                # print(f"Max of optimized grad: {max_optimized_grad}")

                # 把优化后的梯度放入正确的位置
                new_grads_layer[start_index:start_index + param_size] = optimized_grad_layer

                # 更新起始索引
                start_index += param_size
            print(k_sum)
            # 返回全局的梯度张量
            return new_grads_layer  # 如果combine方法中使用了torch.add，这里需要乘以-1

    def apply_cvlc(self, init_value, init_indices, original_grad: torch.tensor,
                   abs_gra, k_max, k_min, s, matrix_top, layer_top, block=1):

        sort_gra, _ = torch.sort(abs_gra, descending=True)
        sort_gra = sort_gra * sort_gra
        norm = torch.sum(sort_gra)
        # norm = torch.sum(topk_values ** 2)  # 使用top-k值的平方和进行归一化

        best_loss = float('inf')
        best_ks = []
        best_bs = []
        best_k = 0

        def dynamic_interval(abs_gra_, k_max_, k_min_):
            # 根据梯度的统计特性调整间隔
            gra_range = abs_gra_.max() - abs_gra_.min()
            base_interval = max(1, int((k_max_ - k_min_) / 18))
            if gra_range > gra_range / 2:
                return max(1, base_interval // 2)  # 减小间隔以提高精度
            elif gra_range <= gra_range / 2:
                return base_interval * 2  # 增大间隔以减少计算量
            return base_interval

        interval = max(1, int((k_max - k_min) / 18))
        packet_min = int(self.packet_size / (18 + s))
        packet_max = int(self.packet_size / (5 + s))

        for k in range(k_max, k_min, -interval):

            # interval = dynamic_interval(abs_gra, k_max, k_min)

            ks = self.generate_packet_sizes(k, packet_min, packet_max, matrix_top, layer_top, block)
            # if ks is None or len(ks) == 0:
            #     break  # 跳出最外层循环

            bs = [int(self.packet_size / ks_i - s) for ks_i in ks]
            if min(bs) < 2:
                continue

            spar_loss = torch.sum(sort_gra[sum(ks):]) / norm
            quan_loss = 0
            accumu_k = 0
            for i in range(len(ks)):
                # indices = topk_indices[accumu_k:accumu_k + ks[i]]
                quan_loss += (self.PQ_loss(ks[i], bs[i], sort_gra[accumu_k:accumu_k + ks[i]]) / norm)
                accumu_k += ks[i]
            loss = spar_loss + quan_loss

            # 迭代优化过程
            last_loss = 0
            for _ in range(20):
                kL = 0
                for i in range(len(ks) - 1):
                    kR = kL + ks[i] + ks[i + 1]
                    x, y = ks[i], ks[i + 1]
                    xb, yb = bs[i], bs[i + 1]
                    fx_loss = (self.PQ_loss(x, xb, sort_gra[kL: kL + x]) +
                               self.PQ_loss(y, yb, sort_gra[kL + x:kR])) / norm

                    for xb in range(2, 21):
                        x = int(self.packet_size / (s + xb))
                        y = kR - kL - x
                        if x <= 0 or y <= 0:
                            continue
                        if y <= s:  # 确保 y 大于 s，否则跳过此次循环
                            continue

                        yb = int(self.packet_size / (y - s))

                        if yb < 2 or yb > 20:
                            continue
                        # tmp_loss = (self.PQ_loss(x, xb, topk_values[topk_indices[kL:kL + x]]) +
                        #             self.PQ_loss(y, yb, topk_values[topk_indices[kL + x:kR]])) / norm
                        tmp_loss = ((self.PQ_loss(x, xb, sort_gra[kL:kL + x])) + (
                            self.PQ_loss(y, yb, sort_gra[kL + x:kR]))) / norm
                        if tmp_loss < fx_loss:
                            fx_loss = tmp_loss
                            ks[i], ks[i + 1] = x, y
                            bs[i], bs[i + 1] = xb, yb
                    kL += ks[i]

                quan_loss = 0
                accumu_k = 0
                for i in range(len(ks)):
                    quan_loss += (self.PQ_loss(ks[i], bs[i], sort_gra[accumu_k:accumu_k + ks[i]]) / norm)
                    accumu_k += ks[i]
                loss = spar_loss + quan_loss

                if loss == last_loss:
                    break
                last_loss = loss

            if loss < best_loss:
                best_loss = loss
                best_k = sum(ks)
                best_ks = ks
                best_bs = bs

        # print("the best ks of layer is ", best_ks)
        # print("the best k of layer is", best_k)
        # print("the best loss of layer is ", best_loss)
        # print("the best bs of layer is", best_bs)

        final_v, final_indices = torch.topk(init_value, best_k)
        final_index = init_indices[final_indices]

        accumu_k_ = 0

        new_grad = torch.zeros_like(original_grad).to(original_grad.device)
        new_grad[final_index] = original_grad[final_index]
        # 我们需要确保best_ks和best_bs已经正确设置，因为它们会在循环中使用。
        for i in range(len(best_ks)):
            # 选择这个部分的索引
            indices = final_index[accumu_k_:accumu_k_ + best_ks[i]]

            # 对选中的梯度值进行量化
            quantized_values = self.PQ_quan(new_grad[indices], 2 ** (best_bs[i] - 1))

            # 在选定的索引处更新new_grad张量
            new_grad[indices] = quantized_values

            # 从原始梯度中减去量化的值
            original_grad[indices] -= new_grad[indices]

            # 累加处理的索引数量
            accumu_k_ += best_ks[i]

        return new_grad, best_k

    def cvlc_block(self, net, last_net, blocks):
        grad = torch.tensor([]).to(self.device)
        param = torch.tensor([]).to(self.device)
        k_sum = 0
        for layer in net.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer:
                param_now = net.state_dict()[layer].detach().to(self.device)
                param_last = last_net.state_dict()[layer].detach().to(self.device)
                param_g = param_now - param_last
                param = torch.cat((param, param_now.view(-1)))
                grad = torch.cat((grad, param_g.view(-1)))
        new_grad = torch.zeros_like(grad).to(self.device)

        position = self.d // blocks
        s = int(math.log(position, 2)) + 1
        kk_max = int(self.packet_size / (s + 5)) * self.packet_num
        kk_min = int(self.packet_size / (s + 18)) * self.packet_num

        k_max = kk_max // blocks
        k_min = kk_min // blocks
        init_value, init_indices, list_grad, list_param = self.topk_block(blocks, grad, param, k_max)
        start_index = 0
        for block in range(blocks):
            param_size = list_param[block].numel()
            optimized_grad, k_best = self.apply_cvlc(init_value[block], init_indices[block], list_grad[block],
                                                     list_grad[block].abs(), k_max, k_min, s, False, False, blocks)
            k_sum += k_best
            new_grad[start_index:start_index + param_size] = optimized_grad

            start_index += param_size
        print(k_sum)
        return new_grad

    def topk_block(self, blocks, grad, param, k):
        len_split = len(grad)
        len_block = len_split // blocks
        list_block = [len_block] * blocks
        if sum(list_block) != len_split:
            list_block[-1] = list_block[-1] + len_split - sum(list_block)

        list_grad = list(torch.split(grad, list_block, dim=0))
        list_param = list(torch.split(param, list_block, dim=0))
        init_value = []
        init_indices = []
        for i in range(blocks):
            if self.gradient:
                i_value, i_indices = torch.topk(list_grad[i].abs(), k=min(k, list_grad[i].numel()), largest=True)
            elif self.graproduct:
                grad_param_product = list_param[i] * list_grad[i]
                i_value, i_indices = torch.topk(grad_param_product.abs(), k=min(k, grad_param_product.numel()),
                                                largest=True)
            elif self.graproduct_2:
                grad_param_product_squared = (list_param[i] * list_grad[i]).pow(2)
                i_value, i_indices = torch.topk(grad_param_product_squared.abs(),
                                                k=min(k, grad_param_product_squared.numel()), largest=True)
            elif self.adalora:
                grad_param_product = list_param[i] * list_grad[i]
                T_current = self.beta_1 * self.T_previous.get(i, torch.zeros_like(grad_param_product)) + (
                        1 - self.beta_1) * grad_param_product
                U_current = self.beta_2 * self.U_previous.get(i, torch.zeros_like(grad_param_product)) + (
                        1 - self.beta_2) * (grad_param_product - T_current).abs()
                self.T_previous[i] = T_current
                self.U_previous[i] = U_current
                importance_score = T_current * U_current
                i_value, i_indices = torch.topk(importance_score.abs(), k=min(k, grad_param_product.numel()),
                                                largest=True)
            else:
                exit(0)

            init_value.append(i_value)
            init_indices.append(i_indices)
        return init_value, init_indices, list_grad, list_param

    def generate_packet_sizes(self, k, packet_min, packet_max, matrix_top, layer_top, block=1):
        if matrix_top:
            block = self.matrix
        elif layer_top:
            layer = self.matrix // 4
            block = layer

        if k < self.packet_num // block * packet_min:
            # return np.array([])
            return np.full(self.packet_num // block, packet_min)
        # 随机初始化数据包大小，保证总和接近但可能不等于k
        sizes = np.random.randint(packet_min, packet_max + 1, size=self.packet_num // block)
        diff = k - sizes.sum()

        # 微调数据包大小以确保总和恰好为k
        while diff != 0:
            # 每次循环进行细微调整
            for i in range(self.packet_num):
                if diff == 0:
                    break

                if diff > 0 and sizes[i] < packet_max:
                    # 总数少了，且当前包未达最大值，可以增加
                    increment = min(packet_max - sizes[i], diff)  # 只增加到最大值或所需的数量
                    sizes[i] += increment
                    diff -= increment
                elif diff < 0 and sizes[i] > packet_min:
                    # 总数多了，且当前包未达最小值，可以减少
                    decrement = min(sizes[i] - packet_min, -diff)  # 只减少到最小值或所需的数量
                    sizes[i] -= decrement
                    diff += decrement

        return sizes

    def generate_packet_sizes_1(self, k, packet_min, packet_max, packet_num):
        if k < packet_num:
            # return np.array([])
            return np.full(packet_num, packet_min)
        # 随机初始化数据包大小，保证总和接近但可能不等于k
        sizes = np.random.randint(packet_min, packet_max + 1, size=packet_num)
        diff = k - sizes.sum()

        # 微调数据包大小以确保总和恰好为k
        while diff != 0:
            # 每次循环进行细微调整
            for i in range(packet_num):
                if diff == 0:
                    break
                if diff > 0 and sizes[i] < packet_max:
                    # 总数少了，且当前包未达最大值，可以增加
                    increment = min(packet_max - sizes[i], diff)  # 只增加到最大值或所需的数量
                    sizes[i] += increment
                    diff -= increment
                elif diff < 0 and sizes[i] > packet_min:
                    # 总数多了，且当前包未达最小值，可以减少
                    decrement = min(sizes[i] - packet_min, -diff)  # 只减少到最小值或所需的数量
                    sizes[i] -= decrement
                    diff += decrement
        return sizes

    def generate_packet_sizes_2(self, k, packet_min, packet_max, matrix_top, layer_top):
        if matrix_top:
            block = self.matrix
        elif layer_top:
            layer = self.matrix // 4
            block = layer
        else:
            block = 1

        if k < self.packet_num // block * packet_min:
            # return np.array([])
            return np.full(self.packet_num // block, packet_min)
        # 初始化数据包大小
        sizes = np.random.randint(packet_min, packet_max + 1, size=self.packet_num // block)
        total = sizes.sum()

        # 微调以确保总和匹配k
        while total != k:
            diff = k - total
            if abs(diff) > self.packet_num:  # 如果差距较大
                step = max(1, abs(diff) // self.packet_num)  # 动态调整步长
            else:
                step = 1  # 接近目标时使用最小步长

            for _ in range(self.packet_num):
                idx = np.random.randint(self.packet_num)  # 随机选择一个包进行调整
                if diff > 0 and sizes[idx] < packet_max:
                    increment = min(step, packet_max - sizes[idx], diff)
                    sizes[idx] += increment
                    total += increment
                    diff -= increment
                    if total == k:
                        break
                elif diff < 0 and sizes[idx] > packet_min:
                    decrement = min(step, sizes[idx] - packet_min, -diff)
                    sizes[idx] -= decrement
                    total -= decrement
                    diff += decrement
                    if total == k:
                        break

        return sizes

    def cvlc(self, client_id, network):
        parameters = torch.tensor([]).to(self.device)
        network.to(self.device)
        for p in network.state_dict():
            if p.find("num_batches_tracked") != -1:
                continue
            if 'lora' in p:
                parameters = torch.cat((parameters, network.state_dict()[p].view(-1).detach()))
        last_parameters = torch.tensor([]).to(self.device)
        last_nework = copy.deepcopy(self.global_net).to(self.device)
        for p in last_nework.state_dict():
            # for p in self.last_model.parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            if 'lora' in p:
                last_parameters = torch.cat((last_parameters, last_nework.state_dict()[p].view(-1).detach()))
        self.accu_gra[client_id] = last_parameters - parameters + self.accu_gra[client_id]

        init_value, init_indices = torch.topk(self.accu_gra[client_id].abs(), self.k_max)
        s = int(math.log(self.d, 2)) + 1

        self.best_k = 0
        self.best_ks = []
        self.best_bs = []
        self.best_loss = float('inf')

        abs_gra = self.accu_gra[client_id].abs()
        sort_gra, _ = torch.sort(abs_gra, descending=True)
        sort_gra = sort_gra * sort_gra
        norm = torch.sum(sort_gra)

        loss_func = self.PQ_loss
        interval = int((self.k_max - self.k_min) / 20)

        packet_min = int(self.packet_size / (20 + s))
        packet_max = int(self.packet_size / (4 + s))

        for k in range(self.k_max, self.k_min, -interval):
            ks = self.generate_packet_sizes_2(k, packet_min, packet_max, matrix_top=False, layer_top=False)
            ks.sort()
            bs = [int(self.packet_size / ks_i - s) for ks_i in ks]
            if bs[0] < 2:
                continue
            # print("the parameter number in the packet is {}, the bit in the packet is {}".format(ks[0], bs[0]))

            #  计算固定k后的稀疏化误差
            spar_loss = torch.sum(sort_gra[sum(ks):]) / norm
            # print("the k is {}, the spar_loss is {}".format(sum(ks), spar_loss))
            quan_loss = 0
            start = 0
            for i in range(self.packet_num):
                quan_loss += (loss_func(ks[i], bs[i], sort_gra[start:start + ks[i]]) / norm)
                start += ks[i]
            loss = spar_loss + quan_loss
            # print("the initial loss is ", loss)

            last_loss = 0
            for _ in range(20):
                kL = 0
                for i in range(self.packet_num - 1):
                    kR = kL + ks[i] + ks[i + 1]
                    x, y = ks[i], ks[i + 1]
                    xb, yb = bs[i], bs[i + 1]
                    fx_loss = (loss_func(x, xb, sort_gra[kL: kL + x]) + loss_func(y, yb, sort_gra[kL + x:kR])) / norm

                    for xb in range(2, 21):
                        x = int(self.packet_size / (s + xb))
                        y = kR - kL - x
                        if x <= 0 or y <= 0:
                            break
                        yb = int(self.packet_size / y - s)

                        if yb < 2 or yb > 20:
                            continue
                        tmp_loss = ((loss_func(x, xb, sort_gra[kL:kL + x])) + (
                            loss_func(y, yb, sort_gra[kL + x:kR]))) / norm
                        if tmp_loss < fx_loss:
                            fx_loss = tmp_loss
                            ks[i], ks[i + 1] = x, y
                            bs[i], bs[i + 1] = xb, yb
                    kL += ks[i]
                quan_loss = 0
                accumu_k = 0
                for i in range(self.packet_num):
                    quan_loss += (loss_func(ks[i], bs[i], sort_gra[accumu_k:accumu_k + ks[i]]) / norm)
                    accumu_k += ks[i]
                loss = spar_loss + quan_loss
                # print("the compressed loss after optimizer is ", loss)
                if loss == last_loss:
                    break
                last_loss = loss
            if loss < self.best_loss:
                self.best_k = sum(ks)
                self.best_ks = ks
                self.best_bs = bs
                self.best_loss = loss
        # print("the best loss is {} when k is {}".format(self.best_loss, self.best_k))
        # print("the best bs is ", self.best_bs)
        # print("the best k is ", self.best_k)
        print(self.best_k)

        final_v, i = torch.topk(init_value, self.best_k)
        final_index = init_indices[i]
        accumu_k = 0
        new_gra = torch.zeros(self.accu_gra[client_id].shape).to(self.device)
        new_gra[final_index] = self.accu_gra[client_id][final_index]

        for i in range(self.packet_num):
            indices = final_index[accumu_k:accumu_k + self.best_ks[i]]
            new_gra[indices] = self.PQ_quan(new_gra[indices], 2 ** (self.best_bs[i]))
            self.accu_gra[client_id][indices] -= new_gra[indices]
            accumu_k += self.best_ks[i]
        # 这里返回 new_gra * -1 是由于 combine方法中使用了torch.add，而cvlc元素代码使用了减法
        return new_gra * -1

    def PQ_quan(self, parameters, centroids):
        ans = torch.zeros_like(parameters).to(self.device)
        p_max = torch.max(parameters)
        p_min = torch.min(parameters)
        interval = (p_max - p_min) / (centroids - 1)
        left = (((parameters - p_min) / interval).int() * interval + p_min).to(self.device)
        right = (left + interval).to(self.device)
        probability = (parameters - left) / (right - left)
        probability[probability < 1e-5] = 0
        seed = torch.rand(parameters.shape).to(self.device)
        ans[seed < probability] = right[seed < probability]
        ans[seed >= probability] = left[seed >= probability]
        return ans

    def combine(self, grad, net):
        current_index = 0
        current_state_dict = net.state_dict()
        for name, param in current_state_dict.items():
            if 'lora' in name:
                param = param.to(self.device)
                numel = param.data.numel()
                size = param.data.size()
                current_state_dict[name] = \
                    torch.add(grad[current_index:current_index + numel].view(size), param.data.detach())
                current_index += numel
        net.load_state_dict(current_state_dict)
        return net

    def fft(self, network, thresholds=0.5):
        parameters = torch.tensor([]).to(self.device)
        net = copy.deepcopy(network)
        net.to(self.device)
        for p in net.state_dict():
            if p.find("num_batches_tracked") != -1:
                continue
            if 'lora' in p:
                param1 = net.state_dict()[p].detach().to(self.device)
                param2 = self.global_net.state_dict()[p].detach().to(self.device)
                param3 = param1 - param2
                fft_result = torch.fft.fft2(param3)
                fft_shifted = torch.fft.fftshift(fft_result).cuda()
                freq_center = (np.array(param3.shape) - 1) / 2
                y, x = torch.meshgrid(torch.arange(param3.shape[0]), torch.arange(param3.shape[1]),
                                      indexing='ij')
                distances = torch.sqrt((x - freq_center[1]) ** 2 + (y - freq_center[0]) ** 2)
                radius = torch.max(distances) * thresholds
                circular_filter = (distances >= radius).float().cuda()
                filtered_fft = fft_shifted * circular_filter

                # 将频谱进行反中心化
                filtered_fft_unshifted = torch.fft.ifftshift(filtered_fft)

                # 对过滤后的频谱进行逆傅里叶变换
                filtered_parameters = torch.fft.ifft2(filtered_fft_unshifted).real
                filtered_parameters = filtered_parameters.contiguous().to(self.device)
                parameters = torch.cat((parameters, filtered_parameters.view(-1)))
        return parameters

    def save_plot(self, data, rounds, label, filename):

        def ensure_dir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, data, label=label, marker='o')
        plt.xlabel('Communication Rounds')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy of {label}')
        plt.legend()
        plt.grid(True)
        ensure_dir('image')  # 确保 images 目录存在
        plt.savefig(f'image/{filename}.png')
        plt.close()

    def cvlc_topk_block(self, net, last_net, blocks):
        grad = torch.tensor([]).to(self.device)
        param = torch.tensor([]).to(self.device)
        k_sum = 0

        # 计算梯度
        for layer in net.state_dict():
            if layer.find("num_batches_tracked") != -1:
                continue
            if 'lora' in layer:
                param_now = net.state_dict()[layer].detach().to(self.device)
                param_last = last_net.state_dict()[layer].detach().to(self.device)
                param_g = param_now - param_last
                param = torch.cat((param, param_now.view(-1)))
                grad = torch.cat((grad, param_g.view(-1)))

        new_grad = torch.zeros_like(grad).to(self.device)

        # 梯度和参数分块
        len_split = len(grad)
        len_block = len_split // blocks
        list_block = [len_block] * blocks
        if sum(list_block) != len_split:
            list_block[-1] = list_block[-1] + len_split - sum(list_block)

        list_grad = list(torch.split(grad, list_block, dim=0))
        list_param = list(torch.split(param, list_block, dim=0))
        position = self.d // blocks
        s = int(math.log(position, 2)) + 1
        k_max = int(self.packet_size / (s + 5)) * self.packet_num
        # 进行topk
        init_value, init_indices = torch.topk(grad.abs(), k_max, largest=True)
        list_value = [[] for _ in range(blocks)]
        list_indices = [[] for _ in range(blocks)]

        # 对topk进行分块
        param_size = list_param[0].numel()
        for i in range(len(init_indices)):
            index = init_indices[i] // param_size
            if index >= blocks:
                index = blocks - 1
            list_value[index].append(init_value[i])
            list_indices[index].append(init_indices[i] - param_size * index)

        # 计算每个包对应的参数
        list_k_max = [len(l) for l in list_indices]
        # 每个包初始分配一个包
        list_packet_num = [1] * len(list_k_max)
        # 每个有数据的块初始分配一个包
        list_packet_num = [0] * len(list_k_max)
        already_assigned_packer_num = 0
        for i in range(len(list_k_max)):
            if list_k_max[i] != 0:
                list_packet_num[i] = 1
                already_assigned_packer_num += 1
        param_per_packet = []
        for i in range(len(list_k_max)):
            if list_packet_num[i] == 0:
                param_per_packet.append(0)
            else:
                param_per_packet.append(list_k_max[i] / list_packet_num[i])
        # 对每个块分配包
        for num in range(blocks, self.packet_num):
            max_index = param_per_packet.index(max(param_per_packet))
            list_packet_num[max_index] += 1
            param_per_packet[max_index] = list_k_max[max_index] / list_packet_num[max_index]
        # print(param_per_packet)
        # count_non_zero = len([num for num in param_per_packet if num != 0])
        # print(count_non_zero)
        # print(list_packet_num)
        # 分块进行cvlc
        start_index = 0
        for block in tqdm(range(blocks)):
            param_size = list_param[block].numel()
            if list_k_max[block] > 0:
                k_min = int(self.packet_size / (s + 18)) * list_packet_num[block]
                optimized_grad, k_best = self.apply_cvlc_topk_block(torch.tensor(list_value[block]),
                                                                    torch.tensor(list_indices[block]),
                                                                    list_grad[block],
                                                                    list_grad[block].abs(), list_k_max[block],
                                                                    k_min, s, list_packet_num[block])
            else:
                optimized_grad = torch.zeros_like(list_param[block])
                k_best = 0
            k_sum += k_best
            new_grad[start_index:start_index + param_size] = optimized_grad

            start_index += param_size
        return new_grad

    def apply_cvlc_topk_block(self, init_value, init_indices, original_grad: torch.tensor,
                              abs_gra, k_max, k_min, s, packet_num):
        sort_gra, _ = torch.sort(abs_gra, descending=True)
        sort_gra = sort_gra * sort_gra
        norm = torch.sum(sort_gra)
        # norm = torch.sum(topk_values ** 2)  # 使用top-k值的平方和进行归一化

        best_loss = float('inf')
        best_ks = []
        best_bs = []
        best_k = 0

        interval = max(1, int((k_max - k_min) / 18))
        packet_min = int(self.packet_size / (20 + s))
        packet_max = int(self.packet_size / (2 + s))

        for k in range(k_max, k_min, -interval):

            ks = self.generate_packet_sizes_1(k, packet_min, packet_max, packet_num)
            # if ks is None or len(ks) == 0:
            #     break  # 跳出最外层循环

            bs = [int(self.packet_size / ks_i - s) for ks_i in ks]
            if min(bs) < 2:
                continue

            spar_loss = torch.sum(sort_gra[sum(ks):]) / norm
            quan_loss = 0
            accumu_k = 0
            for i in range(len(ks)):
                # indices = topk_indices[accumu_k:accumu_k + ks[i]]
                quan_loss += (self.PQ_loss(ks[i], bs[i], sort_gra[accumu_k:accumu_k + ks[i]]) / norm)
                accumu_k += ks[i]
            loss = spar_loss + quan_loss

            # 迭代优化过程
            last_loss = 0
            for _ in range(20):
                kL = 0
                for i in range(len(ks) - 1):
                    kR = kL + ks[i] + ks[i + 1]
                    x, y = ks[i], ks[i + 1]
                    xb, yb = bs[i], bs[i + 1]
                    fx_loss = (self.PQ_loss(x, xb, sort_gra[kL: kL + x]) +
                               self.PQ_loss(y, yb, sort_gra[kL + x:kR])) / norm

                    for xb in range(2, 21):
                        x = int(self.packet_size / (s + xb))
                        y = kR - kL - x
                        if x <= 0 or y <= 0:
                            continue
                        if y <= s:  # 确保 y 大于 s，否则跳过此次循环
                            continue

                        yb = int(self.packet_size / (y - s))

                        if yb < 2 or yb > 20:
                            continue
                        # tmp_loss = (self.PQ_loss(x, xb, topk_values[topk_indices[kL:kL + x]]) +
                        #             self.PQ_loss(y, yb, topk_values[topk_indices[kL + x:kR]])) / norm
                        tmp_loss = ((self.PQ_loss(x, xb, sort_gra[kL:kL + x])) + (
                            self.PQ_loss(y, yb, sort_gra[kL + x:kR]))) / norm
                        if tmp_loss < fx_loss:
                            fx_loss = tmp_loss
                            ks[i], ks[i + 1] = x, y
                            bs[i], bs[i + 1] = xb, yb
                    kL += ks[i]

                quan_loss = 0
                accumu_k = 0
                for i in range(len(ks)):
                    quan_loss += (self.PQ_loss(ks[i], bs[i], sort_gra[accumu_k:accumu_k + ks[i]]) / norm)
                    accumu_k += ks[i]
                loss = spar_loss + quan_loss

                if loss == last_loss:
                    break
                last_loss = loss

            if loss < best_loss:
                best_loss = loss
                best_k = sum(ks)
                best_ks = ks
                best_bs = bs

        # print("the best bs of layer is \n", best_ks)
        # print("the best k of layer is \n", best_k)
        # print("the best loss of layer is\n ", best_loss)
        # print("the best bs of layer is ", best_bs)

        final_v, final_indices = torch.topk(init_value, best_k)
        final_index = init_indices[final_indices]

        accumu_k_ = 0

        new_grad = torch.zeros_like(original_grad).to(original_grad.device)
        new_grad[final_index] = original_grad[final_index]
        # 我们需要确保best_ks和best_bs已经正确设置，因为它们会在循环中使用。
        for i in range(len(best_ks)):
            # 选择这个部分的索引
            indices = final_index[accumu_k_:accumu_k_ + best_ks[i]]

            # 对选中的梯度值进行量化
            quantized_values = self.PQ_quan(new_grad[indices], 2 ** (best_bs[i] - 1))

            # 在选定的索引处更新new_grad张量
            new_grad[indices] = quantized_values

            # 从原始梯度中减去量化的值
            original_grad[indices] -= new_grad[indices]

            # 累加处理的索引数量
            accumu_k_ += best_ks[i]
        return new_grad, best_k

    def run(self):
        """ Start federated learning scenario """
        test_acc_list = []

        self.set_accu_gra()
        for rnd in range(self.args.comm_round):
            cohorts = np.random.choice(self.args.n_parties, int(self.args.n_parties * self.args.sample_fraction),
                                       replace=False).tolist()
            local_nets = {}
            for cidx, client_id in enumerate(cohorts):
                print('client{}'.format(client_id))
                local_nets[client_id] = copy.deepcopy(self.global_net)
                self.training_plm(client_id=client_id, networks=local_nets[client_id])
                local_nets[client_id] = local_nets[client_id].cpu()

                cvlc = False
                if cvlc:
                    print("begin to train cvcl")
                    grad = self.cvlc(client_id, local_nets[client_id])
                    cvlc_net = self.combine(grad, copy.deepcopy(self.global_net))
                    # test_acc_cvcl = self.testing_multi_lang_plm(cvlc_net)
                    # print(test_acc_cvcl)
                    # test_acc_hist_cvcl.append(float(test_acc_cvcl))
                    # 需要使用哪种方法把其下面的代码解除注释
                    local_nets[client_id] = cvlc_net
                fft = False
                if fft:
                    print("begin to train fft")
                    grad = self.fft(local_nets[client_id])
                    fft_net = self.combine(grad, copy.deepcopy(self.global_net))
                    # test_acc_fft = self.testing_multi_lang_plm(fft_net)
                    # print(test_acc_fft)
                    # test_acc_hist_fft.append(float(test_acc_fft))
                    local_nets[client_id] = fft_net
                tk = False
                if tk:
                    print("begin to train topk")
                    grad, mask = self.topk(local_nets[client_id])
                    # if rnd == 5:
                    #     df = pd.DataFrame(mask.cpu())
                    #     df.to_csv('matrix.csv', index=False, header=False)
                    tk_net = self.combine(grad, copy.deepcopy(self.global_net))
                    # test_acc_topk = self.testing_multi_lang_plm(tk_net)
                    # print(test_acc_topk)
                    # test_acc_hist_topk.append(float(test_acc_topk))
                    local_nets[client_id] = tk_net
                cvlc_for_layer = False
                if cvlc_for_layer:
                    print("begin to train cvlc layer")
                    grad = self.cvlc_layer(client_id, local_nets[client_id], layer_top=True, matrix_top=False)
                    cvlc_for_layer_net = self.combine(grad, copy.deepcopy(self.global_net))
                    # test_acc_layer = self.testing_multi_lang_plm(cvlc_for_layer_net)
                    # print(test_acc_layer)
                    # test_acc_hist_layer.append(float(test_acc_layer))
                    local_nets[client_id] = cvlc_for_layer_net
                cvlc_for_matrix = False
                if cvlc_for_matrix:
                    print("begin to train cvlc matrix")
                    grad = self.cvlc_layer(client_id, local_nets[client_id], matrix_top=True, layer_top=False)
                    cvlc_for_matrix_net = self.combine(grad, copy.deepcopy(self.global_net))
                    # test_acc_matrix = self.testing_multi_lang_plm(cvlc_for_matrix_net)
                    # print(test_acc_matrix)
                    local_nets[client_id] = cvlc_for_matrix_net
                cvlc_for_block = False
                if cvlc_for_block:
                    print("begin to train cvlc block")
                    grad = self.cvlc_block(net=local_nets[client_id], last_net=self.global_net, blocks=4)
                    cvlc_for_block_net = self.combine(grad, copy.deepcopy(self.global_net))
                    # test_acc_block = self.testing_multi_lang_plm(cvlc_for_block_net)
                    # print(test_acc_block)
                    local_nets[client_id] = cvlc_for_block_net
                cvlc_topk_block = False
                if cvlc_topk_block:
                    print("begin to train cvlc topk block")
                    grad = self.cvlc_topk_block(net=local_nets[client_id], last_net=self.global_net, blocks=4)
                    cvlc_for_topk_block_net = self.combine(grad, copy.deepcopy(self.global_net))
                    local_nets[client_id] = cvlc_for_topk_block_net
            if self.args.local is False:
                self.aggregate(local_nets, cohorts)
            print('Round ({}/{})'.format(rnd + 1, self.args.comm_round))
            test_acc = self.testing_multi_lang_plm()
            print(test_acc)
            test_acc_list.append(float(test_acc))
            if test_acc >= 0.6:
                break

        with open('./save/cvlc_for_block_4.txt', 'w') as file:
            # 遍历列表中的每个元素，并将其转换为字符串后写入文件
            for item in test_acc_list:
                file.write(str(item) + '\n')
        # plt.figure(figsize=(12, 8))
        # if test_acc_hist_layer:
        #     self.save_plot(test_acc_hist_layer, list(range(1, len(test_acc_hist_layer) + 1)), 'CVLC Layer',
        #                    'cvlc_layer_accuracy')
        # if test_acc_hist_matrix:
        #     self.save_plot(test_acc_hist_matrix, list(range(1, len(test_acc_hist_matrix) + 1)), 'CVLC Matrix',
        #                    'cvlc_matrix_accuracy')
        # if test_acc_hist_cvcl:
        #     self.save_plot(test_acc_hist_cvcl, list(range(1, len(test_acc_hist_cvcl) + 1)), 'CVLC', 'cvlc_accuracy')
        # if test_acc_hist_fft:
        #     self.save_plot(test_acc_hist_fft, list(range(1, len(test_acc_hist_fft) + 1)), 'FFT', 'fft_accuracy')
        # if test_acc_hist_topk:
        #     self.save_plot(test_acc_hist_topk, list(range(1, len(test_acc_hist_topk) + 1)), 'Top-k', 'topk_accuracy')
