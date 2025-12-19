import copy
import math
import torch
import pprint
import numpy as np
import torch.optim as optim
# import matplotlib.pyplot as plt
import time
from torch import nn

from quantization import PQ_loss, PQ_quan, QSGD_quan, QSGD_loss


class Fed_cvlc(object):
    def __init__(self, args, trainable_parameters, matrix_num):
        super().__init__()
        self.args = args
        self.min_bit = 4
        self.max_bit = args.bit
        self.packet_size = 1500 * 8
        self.packet_num = args.packet_num
        self.trainable_parameters = trainable_parameters
        self.matrix_num = matrix_num
        self.new_blocks_size = 0

        self.s = None
        self.d = None
        self.packet_max = None
        self.packet_min = None

        self.beta_1 = 0.85
        self.beta_2 = 0.85
        self.T_previous = {}
        self.U_previous = {}

        self.compression = []
        self.compute = []

    def do_compress(self, grad: torch.tensor, param: torch.tensor, name_grad, name_param, proportion, rnd):
        method = self.args.method
        if method == 'raw':
            return grad
        elif method == 'cvlc':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad = self.cvlc_raw(grad)
        elif method == 'block':
            block = self.args.blocks
            self.d = self.trainable_parameters // block
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad, _ = self.cvlc_block(grad=grad, param=param, blocks=block)
        elif method == 'block_opt':
            block = self.args.blocks
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad, block = self.cvlc_block_optim(grad=grad, param=param, blocks=block, rnd=rnd)
            print("the number and proportion of non-zero is ", torch.count_nonzero(grad), torch.count_nonzero(grad) / grad.numel())
            # with open(f'block{self.args.dataset}_{self.args.subdataset}_r_25.txt', "a") as file:
            #     file.write(f"{block}\n")
        elif method == 'layer':
            block = self.matrix_num // 4
            self.d = self.trainable_parameters // block
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad, _ = self.cvlc_block(grad=grad, param=param, blocks=block)
        elif method == 'matrix':
            block = self.matrix_num
            self.d = self.trainable_parameters // block
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad, _ = self.cvlc_block(grad=grad, param=param, blocks=block)
        elif method == 'topk_block':
            block = self.args.blocks
            self.d = self.trainable_parameters // block
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad = self.cvlc_topk_block(grad=grad, param=param, blocks=block)
        elif method == 'topk':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            # k = int(self.packet_size / (32 + self.s) * self.packet_num)
            k = int(self.args.proportion * self.d)
            grad = self.topk_pure(grad=grad, k=k, name_param=name_param, name_grad=name_grad)
        elif method == 'randk':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            # k = int(self.packet_size / (32 + self.s) * self.packet_num)
            k = int(0.01 * self.d)
            grad = randk(grad=grad, k=k)
        elif method == 'topk_AB':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            grad = self.topk_AB(proportion, name_grad, grad)
        elif method == 'signSGD':
            grad = self.signSGD(grad)
        elif method == 'randk_AB':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            grad = randk_AB(0.01, name_grad, name_param)
        elif method == 'new':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            grad = new(0.01, name_grad, name_param)
        elif method == 'topk_rowcol':
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            grad = topk_rowcol(0.1, name_grad)
        elif method == 'topk_pq':
            block = self.args.blocks
            self.d = self.trainable_parameters // block
            self.s = math.ceil(math.log(self.d, 2))
            bit_len = self.args.bit
            self.packet_min = int(self.packet_size / (bit_len + self.s))
            self.packet_max = self.packet_min
            grad = self.topk_pq(grad=grad, param=param, blocks=block, bit_len=bit_len)
        elif method == 'pq':
            grad = pq_pure(grad=grad, bit_len=6)
        elif method == 'test':
            s1 = None
            loss = []
            for block in range(1, 73):
                self.d = self.trainable_parameters // block
                self.s = math.ceil(math.log(self.d, 2) + 0.1)
                self.packet_min = int(self.packet_size / (self.max_bit + self.s))
                self.packet_max = int(self.packet_size / (self.min_bit + self.s))
                s1, k = self.cvlc_block(grad=grad, param=param, blocks=block)
                print(block, self.s, k.item())
                loss.append(k.item())

            grad = s1
            with open('loss.txt', 'w') as file:
                for item in loss:
                    file.write(str(item) + '\n')
            # exit(0)
        else:
            exit('No method {} erro in do_compress'.format(method))
        return grad

    
    def generate_packet_sizes(self, k, packet_num):
        if k < packet_num:
            return np.full(packet_num, self.packet_min)
        # 随机初始化数据包大小，保证总和接近但可能不等于k
        sizes = np.random.randint(self.packet_min, self.packet_max + 1, size=packet_num)
        diff = k - sizes.sum()

        # 微调数据包大小以确保总和恰好为k
        while diff != 0:
            # 每次循环进行细微调整
            for i in range(packet_num):
                if diff == 0:
                    break
                if diff > 0 and sizes[i] < self.packet_max:
                    # 总数少了，且当前包未达最大值，可以增加
                    increment = min(self.packet_max - sizes[i], diff)  # 只增加到最大值或所需的数量
                    sizes[i] += increment
                    diff -= increment
                elif diff < 0 and sizes[i] > self.packet_min:
                    # 总数多了，且当前包未达最小值，可以减少
                    decrement = min(sizes[i] - self.packet_min, -diff)  # 只减少到最小值或所需的数量
                    sizes[i] -= decrement
                    diff += decrement
        return sizes

    def generate_packet_sizes_2(self, k, packet_num):
        if k < packet_num * self.packet_min:
            # return np.array([])
            return np.full(packet_num, self.packet_min)
        # 初始化数据包大小
        sizes = np.random.randint(self.packet_min, self.packet_max + 1, size=packet_num)
        total = sizes.sum()

        # 微调以确保总和匹配k
        while total != k:
            diff = k - total
            if abs(diff) > packet_num:  # 如果差距较大
                step = max(1, abs(diff) // packet_num)  # 动态调整步长
            else:
                step = 1  # 接近目标时使用最小步长
            for _ in range(packet_num):
                idx = np.random.randint(packet_num)  # 随机选择一个包进行调整
                if diff > 0 and sizes[idx] < self.packet_max:
                    increment = min(step, self.packet_max - sizes[idx], diff)
                    sizes[idx] += increment
                    total += increment
                    diff -= increment
                    if total == k:
                        break
                elif diff < 0 and sizes[idx] > self.packet_min:
                    decrement = min(step, sizes[idx] - self.packet_min, -diff)
                    sizes[idx] -= decrement
                    total -= decrement
                    diff += decrement
                    if total == k:
                        break
        return sizes

    def cvlc_raw(self, grad):
        best_k = 0
        best_ks = []
        best_bs = []
        best_loss = float('inf')

        k_max = self.packet_max * self.packet_num
        k_min = self.packet_min * self.packet_num

        abs_gra = grad.abs()
        sort_gra, _ = torch.sort(abs_gra, descending=True)
        sort_gra = sort_gra * sort_gra
        norm = torch.sum(sort_gra)

        init_value, init_indices = torch.topk(abs_gra, k_max)

        loss_func = PQ_loss
        interval = int((k_max - k_min) / self.max_bit)

        for k in range(k_max, k_min, -interval):
            ks = self.generate_packet_sizes(k, self.packet_num)
            ks.sort()
            bs = [int(self.packet_size / ks_i - self.s) for ks_i in ks]
            if bs[0] < self.min_bit:
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
            # print(spar_loss, quan_loss)
            # print("the initial loss is ", loss)

            last_loss = 0
            for _ in range(self.max_bit):
                kL = 0
                for i in range(self.packet_num - 1):
                    kR = kL + ks[i] + ks[i + 1]
                    x, y = ks[i], ks[i + 1]
                    xb, yb = bs[i], bs[i + 1]
                    fx_loss = (loss_func(x, xb, sort_gra[kL: kL + x]) + loss_func(y, yb, sort_gra[kL + x:kR])) / norm

                    for xb in range(self.min_bit, self.max_bit):
                        x = int(self.packet_size / (self.s + xb))
                        y = kR - kL - x
                        if x <= 0 or y <= 0:
                            break
                        yb = int(self.packet_size / y - self.s)

                        if yb < self.min_bit or yb > self.max_bit:
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
            if loss < best_loss:
                best_k = sum(ks)
                best_ks = ks
                best_bs = bs
                best_loss = loss
        # print("the best loss is {} when k is {}".format(best_loss, best_k))
        # print(self.s)
        # print("the best bs is ", best_bs)
        # print("the best ks is ", best_ks)
        # print("totally k is ", sum(best_ks))
        # print("totally k is ", best_k)

        final_v, i = torch.topk(init_value, best_k)
        final_index = init_indices[i]
        accumu_k = 0
        new_grad = torch.zeros_like(grad)
        new_grad[final_index] = grad[final_index]

        for i in range(self.packet_num):
            indices = final_index[accumu_k:accumu_k + best_ks[i]]
            new_grad[indices] = PQ_quan(new_grad[indices], 2 ** (best_bs[i] - 1))
            accumu_k += best_ks[i]
        return new_grad

    def apply_cvlc_vary(self, init_value, init_indices: torch.tensor, grad: torch.tensor, packet_num,
                        norm, packet_max, packet_min, ss):
        abs_grad = grad.abs()
        sort_grad, _ = torch.sort(abs_grad, descending=True)
        sort_grad = sort_grad * sort_grad

        # 这个可能有问题，norm应该是所有块的，而不是单独这一个块的。
        # norm = torch.sum(sort_grad)

        best_loss = float('inf')
        best_ks = []
        best_bs = []
        best_k = 0
        # 修改这里，
        k_max = packet_max * packet_num
        k_min = packet_min * packet_num

        # def dynamic_interval(abs_gra_, k_max_, k_min_):
        #     # 根据梯度的统计特性调整间隔
        #     gra_range = abs_gra_.max() - abs_gra_.min()
        #     base_interval = max(1, int((k_max_ - k_min_) / 18))
        #     if gra_range > gra_range / 2:
        #         return max(1, base_interval // 2)  # 减小间隔以提高精度
        #     elif gra_range <= gra_range / 2:
        #         return base_interval * 2  # 增大间隔以减少计算量
        #     return base_interval

        interval = max(1, int((k_max - k_min) / self.max_bit))

        for k in range(k_max, k_min, -interval):
            ks = self.generate_packet_sizes(k, packet_num)
            ks.sort()
            bs = [int(self.packet_size / ks_i - ss) for ks_i in ks]

            if bs[0] < self.min_bit:
                continue

            spar_loss = torch.sum(sort_grad[sum(ks):]) / norm
            quan_loss = 0
            accumu_k = 0
            for i in range(len(ks)):
                # indices = topk_indices[accumu_k:accumu_k + ks[i]]
                quan_loss += (PQ_loss(ks[i], bs[i], sort_grad[accumu_k:accumu_k + ks[i]]) / norm)
                accumu_k += ks[i]
            loss = spar_loss + quan_loss

            # 迭代优化过程
            last_loss = 0
            for _ in range(self.max_bit):
                kL = 0
                for i in range(len(ks) - 1):
                    kR = kL + ks[i] + ks[i + 1]
                    x, y = ks[i], ks[i + 1]
                    xb, yb = bs[i], bs[i + 1]
                    fx_loss = (PQ_loss(x, xb, sort_grad[kL: kL + x]) +
                               PQ_loss(y, yb, sort_grad[kL + x:kR])) / norm

                    for xb in range(self.min_bit, self.max_bit):
                        x = int(self.packet_size / (ss + xb))
                        y = kR - kL - x
                        if x <= 0 or y <= 0:
                            continue
                        if y <= ss:  # 确保 y 大于 s，否则跳过此次循环
                            continue

                        yb = int(self.packet_size / y - ss)

                        if yb < self.min_bit or yb > self.max_bit:
                            continue

                        tmp_loss = ((PQ_loss(x, xb, sort_grad[kL:kL + x])) + (
                            PQ_loss(y, yb, sort_grad[kL + x:kR]))) / norm

                        if tmp_loss < fx_loss:
                            fx_loss = tmp_loss
                            ks[i], ks[i + 1] = x, y
                            bs[i], bs[i + 1] = xb, yb
                    kL += ks[i]

                quan_loss = 0
                accumu_k = 0
                for i in range(len(ks)):
                    quan_loss += (PQ_loss(ks[i], bs[i], sort_grad[accumu_k:accumu_k + ks[i]]) / norm)
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

        final_v, final_indices = torch.topk(init_value, best_k)
        final_index = init_indices[final_indices]

        accumu_k_ = 0

        new_grad = torch.zeros_like(grad)
        new_grad[final_index] = grad[final_index]

        for i in range(len(best_ks)):
            # 选择这个部分的索引
            indices = final_index[accumu_k_:accumu_k_ + best_ks[i]]

            # 对选中的梯度值进行量化
            quantized_values = PQ_quan(new_grad[indices], 2 ** (best_bs[i] - 1))

            # 在选定的索引处更新new_grad张量
            new_grad[indices] = quantized_values

            # 累加处理的索引数量
            accumu_k_ += best_ks[i]

        return new_grad

    def apply_cvlc(self, init_value, init_indices: torch.tensor, grad: torch.tensor, packet_num, norm, rnd, block):
        if rnd == 0 or rnd + 1 == self.args.point:
            abs_grad = grad.abs()
            sort_grad, _ = torch.sort(abs_grad, descending=True)
            sort_grad = sort_grad * sort_grad
            # 这个可能有问题，norm应该是所有块的，而不是单独这一个块的。
            # norm = torch.sum(sort_grad)

            best_loss = float('inf')
            best_ks = []
            best_bs = []
            best_k = 0
            # 修改这里，
            k_max = self.packet_max * packet_num
            k_min = self.packet_min * packet_num

            # def dynamic_interval(abs_gra_, k_max_, k_min_):
            #     # 根据梯度的统计特性调整间隔
            #     gra_range = abs_gra_.max() - abs_gra_.min()
            #     base_interval = max(1, int((k_max_ - k_min_) / 18))
            #     if gra_range > gra_range / 2:
            #         return max(1, base_interval // 2)  # 减小间隔以提高精度
            #     elif gra_range <= gra_range / 2:
            #         return base_interval * 2  # 增大间隔以减少计算量
            #     return base_interval

            interval = max(1, int((k_max - k_min) / self.max_bit))

            for k in range(k_max, k_min, -interval):
                ks = self.generate_packet_sizes(k, packet_num)
                ks.sort()
                bs = [int(self.packet_size / ks_i - self.s) for ks_i in ks]

                if bs[0] < self.min_bit:
                    continue

                spar_loss = torch.sum(sort_grad[sum(ks):]) / norm
                quan_loss = 0
                accumu_k = 0
                for i in range(len(ks)):
                    # indices = topk_indices[accumu_k:accumu_k + ks[i]]
                    quan_loss += (PQ_loss(ks[i], bs[i], sort_grad[accumu_k:accumu_k + ks[i]]) / norm)
                    accumu_k += ks[i]
                loss = spar_loss + quan_loss

                # 迭代优化过程
                last_loss = 0
                for _ in range(self.max_bit):
                    kL = 0
                    for i in range(len(ks) - 1):
                        kR = kL + ks[i] + ks[i + 1]
                        x, y = ks[i], ks[i + 1]
                        xb, yb = bs[i], bs[i + 1]
                        fx_loss = (PQ_loss(x, xb, sort_grad[kL: kL + x]) +
                                PQ_loss(y, yb, sort_grad[kL + x:kR])) / norm

                        for xb in range(self.min_bit, self.max_bit):
                            x = int(self.packet_size / (self.s + xb))
                            y = kR - kL - x
                            if x <= 0 or y <= 0:
                                continue
                            if y <= self.s:  # 确保 y 大于 s，否则跳过此次循环
                                continue

                            yb = int(self.packet_size / y - self.s)

                            if yb < self.min_bit or yb > self.max_bit:
                                continue

                            tmp_loss = ((PQ_loss(x, xb, sort_grad[kL:kL + x])) + (
                                PQ_loss(y, yb, sort_grad[kL + x:kR]))) / norm

                            if tmp_loss < fx_loss:
                                fx_loss = tmp_loss
                                ks[i], ks[i + 1] = x, y
                                bs[i], bs[i + 1] = xb, yb
                        kL += ks[i]

                    quan_loss = 0
                    accumu_k = 0
                    for i in range(len(ks)):
                        quan_loss += (PQ_loss(ks[i], bs[i], sort_grad[accumu_k:accumu_k + ks[i]]) / norm)
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

            # print(self.s)
            # print("the best ks of layer is \n", best_ks)
            # print("the best k of layer is \n", best_k)
            # print("the best k of layer is \n", sum(best_ks))
            # print("the best loss of layer is\n ", best_loss)
            print("the best bs of layer is ", best_bs)
            self.best_k[block] = best_k
            self.best_ks[block] = best_ks
            self.best_bs[block] = best_bs
            self.best_loss[block] = best_loss
        

        final_v, final_indices = torch.topk(init_value, self.best_k[block])
        final_index = init_indices[final_indices]

        accumu_k_ = 0

        new_grad = torch.zeros_like(grad)
        new_grad[final_index] = grad[final_index]

        for i in range(len(self.best_ks[block])):
            # 选择这个部分的索引
            indices = final_index[accumu_k_:accumu_k_ + self.best_ks[block][i]]

            # 对选中的梯度值进行量化
            quantized_values = PQ_quan(new_grad[indices], 2 ** (self.best_bs[block][i]))

            # 在选定的索引处更新new_grad张量
            new_grad[indices] = quantized_values

            # 累加处理的索引数量
            accumu_k_ += self.best_ks[block][i]

        return new_grad, self.best_loss[block]

    def topk_grad(self, blocks, grad, param):
        len_split = len(grad)
        len_block = len_split // blocks
        list_block = [len_block] * blocks
        if sum(list_block) != len_split:
            left = len_split - sum(list_block)
            list_block[-left:] = [x + 1 for x in list_block[-left:]]
            # list_block[-1] = list_block[-1] + len_split - sum(list_block)

        list_grad = list(torch.split(grad, list_block, dim=0))
        list_param = list(torch.split(param, list_block, dim=0))
        init_value = []
        init_indices = []

        len_packet_num = self.packet_num // blocks
        list_packet_num = [len_packet_num] * blocks
        if sum(list_packet_num) != self.packet_num:
            left = self.packet_num - sum(list_packet_num)
            list_packet_num[-left:] = [x + 1 for x in list_packet_num[-left:]]

        list_k = [x * self.packet_max for x in list_packet_num]

        for i in range(blocks):
            if self.args.topk_method == 'gradient':
                i_value, i_indices = torch.topk(list_grad[i].abs(), k=min(list_k[i], list_grad[i].numel()),
                                                largest=True)
            elif self.args.topk_method == 'graproduct':
                grad_param_product = list_param[i] * list_grad[i]
                i_value, i_indices = torch.topk(grad_param_product.abs(), k=min(list_k[i], grad_param_product.numel()),
                                                largest=True)
            elif self.args.topk_method == 'graproduct_2':
                grad_param_product_squared = (list_param[i] * list_grad[i]).pow(2)
                i_value, i_indices = torch.topk(grad_param_product_squared.abs(),
                                                k=min(list_k[i], grad_param_product_squared.numel()), largest=True)
            elif self.args.topk_method == 'adalora':
                grad_param_product = list_param[i] * list_grad[i]
                T_current = self.beta_1 * self.T_previous.get(i, torch.zeros_like(grad_param_product)) + (
                        1 - self.beta_1) * grad_param_product
                U_current = self.beta_2 * self.U_previous.get(i, torch.zeros_like(grad_param_product)) + (
                        1 - self.beta_2) * (grad_param_product - T_current).abs()
                self.T_previous[i] = T_current
                self.U_previous[i] = U_current
                importance_score = T_current * U_current
                i_value, i_indices = torch.topk(importance_score.abs(), k=min(list_k[i], grad_param_product.numel()),
                                                largest=True)
            else:
                exit('No topk method {} erro in topk_grad'.format(self.args.topk_method))

            init_value.append(i_value)
            init_indices.append(i_indices)
        # 打印 list_grad 以确认其内容
        # for i, g in enumerate(list_grad):
        #     print(f"list_grad[{i}]: requires_grad={g.requires_grad}, shape={g.shape}")

        return init_value, init_indices, list_grad, list_param, list_packet_num

    def calculate_norm(self, list_grad):
        sort_all_grads, _ = torch.sort(list_grad, descending=True)
        sort_all_grads = sort_all_grads * sort_all_grads
        norm = torch.sum(sort_all_grads)
        return norm

    def ensure_tensor(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], dtype=torch.float32)
        return value

    def calculate_sparse_error(self, list_grad, block_size, norm, packet_num, k):
        # block_size = self.ensure_tensor(block_size)
        # packet_num = self.ensure_tensor(packet_num)
        # avg_r = packet_num / (self.d / block_size)
        # k = int((self.packet_max + self.packet_min) / 2)
        # index = int(avg_r * k)
        # index = k

        sparse_errors = []

        for idx, grad in enumerate(list_grad):
            index = k[idx].size(0)
            sort_grad, _ = torch.sort(grad.abs(), descending=True)
            sort_grad = sort_grad * sort_grad

            sparse_error = torch.sum(sort_grad[index:]) / norm
            sparse_errors.append(sparse_error)

        return sparse_errors

    def calculate_quan_error(self, list_grad, block_size, norm, packet_num, value_bit):
        block_size = self.ensure_tensor(block_size)
        packet_num = self.ensure_tensor(packet_num)
        avg_r = packet_num / (self.d / block_size)
        k = int((self.packet_max + self.packet_min) / 2)

        quan_errors = []

        for grad in list_grad:
            sort_grad, _ = torch.sort(grad.abs(), descending=True)
            sort_grad = sort_grad * sort_grad
            # a = PQ_loss(k, 4, sort_grad[:int(k)])
            # print(a)
            quan_error = avg_r * (PQ_loss(k, value_bit, sort_grad[:int(k)]) / norm)
            quan_errors.append(quan_error)

        return quan_errors

    def block_optim_only(self, grad, param, block_size, norm, value_bit):
        # k_max = 0
        k_min = self.packet_size / (np.log2(self.d) + 32)
        block_size = torch.tensor([block_size], dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([block_size], lr=100)
        min_block_size = max(self.d / self.packet_num, k_min)
        max_iter = 3
        loss_values = []  # 用于存储每次迭代的损失值
        block_num = []  # 用于存储每次迭代的块大小
        spar_errors = []
        quans_errors = []
        last_block = 0
        min_loss = float('inf')
        # min_loss_index = -1
        best_block_size = -1

        for i in range(max_iter):
            print("the i is ", i)
            optimizer.zero_grad()
            bs = block_size.detach().item()
            self.packet_max = int(self.packet_size / (value_bit + 1 + int(math.log(bs, 2))))
            block = int(self.d / block_size)
            init_value, _, list_grad, _, list_packet_num = self.topk_grad(block, grad, param)
            # 确保 grad 和 param 需要梯度
            if not grad.requires_grad:
                grad.requires_grad_(True)
            if not param.requires_grad:
                param.requires_grad_(True)

            list_grad = [g.clone().detach().requires_grad_(True) for g in list_grad]
            
            # 计算误差
            sparse_errors = self.calculate_sparse_error(list_grad, block_size, norm, self.packet_num, init_value)
            quan_errors = self.calculate_quan_error(list_grad, block_size, norm, self.packet_num, value_bit)

            sparse_error = sum(sparse_errors)
            quan_error = sum(quan_errors)

            spar_errors.append(sparse_error.detach().item())
            quans_errors.append(quan_error.detach().item())

            loss = sparse_error + quan_error
            # loss = 0.3 * sparse_error + 0.7 * quan_error
            if not loss.requires_grad:
                loss.requires_grad_(True)

            loss.backward(retain_graph=True)
            optimizer.step()

            # 保证块数是正整数
            with torch.no_grad():
                block_size.clamp_(min=min_block_size)

            loss_values.append(loss.item())
            block_num.append(int(self.d / block_size))

            if loss.item() < min_loss:
                min_loss = loss.item()
                # min_loss_index = i
                best_block_size = copy.deepcopy(block_size.detach().item())
            # if i % 100 == 0:
            #     pprint.pprint(math.ceil(math.log(bs, 2)))
            #     pprint.pprint(f"Iteration {i}: sparse_error={sparse_error.item()}, "
            #                   f"quan_error={quan_error.item()}, loss={loss.item()}")
            #     pprint.pprint("the block_size is {}".format(block_size.item()))
            #     pprint.pprint("-----block_optim-----")

            if bs == last_block:
                # pprint.pprint(math.ceil(math.log(bs, 2)))
                # pprint.pprint(f"Iteration {i}: sparse_error={sparse_error.item()}, "
                #               f"quan_error={quan_error.item()}, loss={loss.item()}")
                # pprint.pprint("the block_size is {}".format(block_size.item()))
                # pprint.pprint("-----block_optim-----")
                break
            last_block = bs
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(max_iter), loss_values, label='Loss')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Loss vs Iteration')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # # 绘制 Sparse Error 图形
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(max_iter), spar_errors, label='Sparse Error', color='orange')
        # plt.xlabel('Iteration')
        # plt.ylabel('Sparse Error')
        # plt.title('Sparse Error vs Iteration')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # # 绘制 Quantization Error 图形
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(max_iter), quans_errors, label='Quantization Error', color='green')
        # plt.xlabel('Iteration')
        # plt.ylabel('Quantization Error')
        # plt.title('Quantization Error vs Iteration')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return round(best_block_size)

    # def packet_num_optim(self, grad, param, new_blocks, list_packet_num, norm, value_bit):
    #     for block in range(new_blocks):
    #         packet_num = torch.tensor([list_packet_num[block]], dtype=torch.float32, requires_grad=True)
    #         optimizer = optim.Adam([packet_num], lr=0.1)
    #         max_iter = 100
    #         loss_values = []

    #         for i in range(max_iter):
    #             optimizer.zero_grad()
    #             grad_block = [grad[block].clone().detach().requires_grad_(True)]

    #             sparse_errors = self.calculate_sparse_error(grad_block, self.d / new_blocks, norm, packet_num, [1])
    #             quan_errors = self.calculate_quan_error(grad_block, self.d / new_blocks, norm, packet_num, value_bit)

    #             sparse_error = sum(sparse_errors)
    #             quan_error = sum(quan_errors)

    #             loss = sparse_error + quan_error
    #             loss.backward()

    #             optimizer.step()
    #             loss_values.append(loss.item())
    #             if i % 10 == 0:
    #                 pprint.pprint(f"Iteration {i}: sparse_error={sparse_error.item()}, ")
    #                 pprint.pprint(f"quan_error={quan_error.item()}, loss={loss.item()}")
    #                 pprint.pprint(f"packet_num={packet_num.item()}")
    #                 pprint.pprint("-----packet_num_optim------")

    #             # 使用 detach() 生成新的 Tensor，然后再执行 clamping 操作
    #             with torch.no_grad():
    #                 packet_num.copy_(packet_num.clamp(min=1))

    #         list_packet_num[block] = round(packet_num.item())
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(range(max_iter), loss_values, label='Loss')
    #         plt.xlabel('Iteration')
    #         plt.ylabel('Loss')
    #         plt.title('Loss vs Iteration')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
    #         print("best packet_num is {}".format(list_packet_num[block]))

    #     return list_packet_num

    def cvlc_block_optim(self, grad, param, blocks, rnd):
        new_grad = torch.zeros_like(grad)
        block_size = self.d / blocks
        norm = self.calculate_norm(grad)
        value_bit = 10
        # 固定k, 优化b和r
        if rnd == 0 or rnd + 1 == self.args.point:
            self.new_blocks_size = self.block_optim_only(grad, param, block_size, norm, value_bit)
        new_block = round(self.d / self.new_blocks_size)
        print("new block is ", new_block)
        self.d = self.trainable_parameters // new_block
        self.s = math.ceil(math.log(self.d, 2))
        self.packet_min = int(self.packet_size / (self.max_bit + self.s))
        self.packet_max = int(self.packet_size / (self.min_bit + self.s))

        if rnd == 0 or rnd + 1 == self.args.point:
            self.best_loss = [float('inf')] * new_block
            self.best_ks = [[]] * new_block
            self.best_bs = [[]] * new_block
            self.best_k = [0] * new_block

        start_index = 0

        # 固定r,k来优化b
        init_value, init_indices, list_grad, list_param, list_packet_num = self.topk_grad(new_block, grad, param)
        print(list_packet_num)
        print(sum(list_packet_num))
        # list_packet_num_pre = list_packet_num.copy()

        # 固定b,k来优化r。发现：稀疏误差占比确实非常大，其实不优化r最好，这样r是均分的，也就是最大的。
        # 如果优化r，因为r在下降，稀疏误差在上升。反而loss在上升
        # 因此直接感觉直接优化b，找到最佳的b，然后直接均分最好。可以在后续消融实验中说明均分更好的地方
        # print("Before packet_num_optim, the list_packet_num is ", list_packet_num_pre)
        #
        # list_packet_num = self.packet_num_optim(list_grad, param, new_blocks, list_packet_num, norm, value_bit)
        #
        # print("After packet_num_optim, the list_packet_num is ", list_packet_num)

        # if sum(list_packet_num) != sum(list_packet_num_pre):
        #     diff = sum(list_packet_num_pre) - sum(list_packet_num)
        #     num_blocks = len(list_packet_num)
        #     distribute = diff // num_blocks
        #     remainder = diff % num_blocks
        #
        #     list_packet_num = [num + distribute for num in list_packet_num]
        #     for i in range(remainder):
        #         list_packet_num[i] += 1

        # 固定b, r优化k
        for block in range(new_block):
            print(block, list_packet_num[block])
            param_size = list_param[block].numel()
            # optimized_grad = self.apply_cvlc_vary(init_value[block], init_indices[block], list_grad[block],
            #                                       list_packet_num[block], norm, packet_min, packet_max, ss)
            optimized_grad, _ = self.apply_cvlc(init_value[block], init_indices[block], list_grad[block],
                                                list_packet_num[block], norm, rnd, block)

            new_grad[start_index:start_index + param_size] = optimized_grad

            start_index += param_size
        return new_grad, new_block

    def cvlc_block(self, grad, param, blocks):
        new_grad = torch.zeros_like(grad)
        # k_max = self.packet_max * self.packet_num
        init_value, init_indices, list_grad, list_param, list_packet_num = self.topk_grad(blocks, grad, param)
        start_index = 0
        k_sum = 0
        norm = self.calculate_norm(grad)

        for block in range(blocks):
            param_size = list_param[block].numel()
            optimized_grad, k = self.apply_cvlc(init_value[block], init_indices[block], list_grad[block],
                                                list_packet_num[block], norm)

            new_grad[start_index:start_index + param_size] = optimized_grad

            start_index += param_size
            k_sum += k
        return new_grad, k_sum

    def cvlc_topk_block(self, grad, param, blocks):
        new_grad = torch.zeros_like(grad)
        # k_max = self.packet_max * self.packet_num
        # 梯度和参数分块
        len_split = len(grad)
        len_block = len_split // blocks
        list_block = [len_block] * blocks
        if sum(list_block) != len_split:
            list_block[-1] = list_block[-1] + len_split - sum(list_block)

        list_grad = list(torch.split(grad, list_block, dim=0))

        init_value_list, init_indices_list, _, _, _ = self.topk_grad(blocks=1, grad=grad.abs(), param=param)
        init_value = init_value_list[0]
        init_indices = init_indices_list[0]

        all_grads = torch.cat([g.abs() for g in list_grad])
        sort_all_grads, _ = torch.sort(all_grads, descending=True)
        sort_all_grads = sort_all_grads * sort_all_grads
        norm = torch.sum(sort_all_grads)

        # init_value, init_indices = torch.topk(grad.abs(), k_max, largest=True)
        list_value = [[] for _ in range(blocks)]
        list_indices = [[] for _ in range(blocks)]

        param_size = list_grad[0].numel()
        for i in range(len(init_indices)):
            index = init_indices[i] // param_size
            if index >= blocks:
                index = blocks - 1
            list_value[index].append(init_value[i])
            list_indices[index].append(init_indices[i] - param_size * index)

        # 计算每个包对应的参数
        list_k_max = [len(indices) for indices in list_indices]

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

        start_index = 0
        for block in range(blocks):
            param_size = list_grad[block].numel()
            if list_k_max[block] > 0:
                optimized_grad, _ = self.apply_cvlc(torch.tensor(list_value[block]), torch.tensor(list_indices[block]),
                                                    list_grad[block], list_packet_num[block], norm)

            else:
                optimized_grad = torch.zeros_like(list_grad[block])

            new_grad[start_index:start_index + param_size] = optimized_grad
            start_index += param_size
        return new_grad

    def topk_pq(self, grad, param, blocks, bit_len):
        new_grad = torch.zeros_like(grad)
        init_value, init_indices, list_grad, list_param, _ = self.topk_grad(blocks, grad, param)
        start_index = 0
        for block in range(blocks):
            param_size = list_param[block].numel()
            optimized_grad = torch.zeros_like(list_grad[block])
            optimized_grad[init_indices[block]] = grad[init_indices[block] + start_index]
            quantized_values = PQ_quan(optimized_grad[init_indices[block]], 2 ** (bit_len - 1))
            optimized_grad[init_indices[block]] = quantized_values
            new_grad[start_index:start_index + param_size] = optimized_grad
            start_index += param_size
        return new_grad


    def topk_pure(self, grad, k, name_param, name_grad):
        """ Perform a topk operation on the gradient and return the processed gradient"""
        grad = torch.tensor([])
        for key in name_grad.keys():
            grad = torch.cat((grad, name_grad[key].view(-1)))
        value, indices = torch.topk(grad.abs(), k)
        mask = torch.zeros_like(grad)
        mask[indices] = 1
        grad = grad * mask
        return grad
    
    def paramTopK(self, name_param, proportion):
        grad = torch.tensor([])
        for key in name_param.keys():
            param = name_param[key].view(-1)
            k = int(proportion * param.numel())
            _, i = torch.topk(param.abs(), k)
            mask = torch.zeros(param.numel())
            mask[i] = 1
            grad = torch.cat((grad, param * mask))
        return grad
    
    def new(self, name_param, proportion):
        index = 0
        grad = torch.tensor([])
        for key in name_param.keys():
            if index % 2 == 0:
                lora_A = name_param[key]
            else:
                lora_B = name_param[key]
                result = (lora_B @ lora_A).view(-1)
                k = int((int(lora_B.numel() * proportion) * (32 + math.log(lora_B.numel(), 2)) + \
                          int(lora_A.numel() * proportion) * (32 + math.log(lora_A.numel(), 2))) / \
                            (32 + math.log(result.numel(), 2)))
                print("the number of uploaded param is ", k)
                print("the origin number of uploaded param is ", 2 * int(lora_B.numel() * proportion))
                print("the shape of loraA and loraB are ", lora_A.shape, lora_B.shape)
                mask = torch.zeros(result.numel())
                _, i = torch.topk(result.abs(), k)
                mask[i] = 1
                grad = torch.cat((grad, result * mask))
            index += 1
        return grad
    
    def prune(self, grad, mask):
        new_grad = grad * mask
        print("the proportion of nz is ", torch.sum(new_grad != 0) / new_grad.numel())
        return new_grad
    
    def CGFedLLM(self, name_grad, autoencoderA, autoencoderB, meanA, stdA, meanB, stdB):
        grad = torch.tensor([])
        real_grad = torch.tensor([])
        for name in name_grad.keys():
            if 'lora_A' in name:
                loraA = name_grad[name]
                data = (loraA.view(1, -1) - meanA) / stdA
                real_grad = torch.cat((real_grad, loraA.view(-1)))
                grad = torch.cat((grad, (autoencoderA(data)).squeeze() * stdA + meanA))
            elif 'lora_B' in name:
                loraB = name_grad[name]
                data = (loraB.view(1, -1) - meanB) / stdB
                real_grad = torch.cat((real_grad, loraB.view(-1)))
                grad = torch.cat((grad, (autoencoderB(data)).squeeze() * stdB + meanB))
        print("the different of real and reconstruct is ", 
              torch.norm(real_grad - grad), torch.norm(real_grad), torch.norm(grad))
        print(grad)
        return grad
    
    def signSGD(self, grad):
        return torch.sign(grad)
    
    def STopK(self, param, grad, name_param, name_grad, proportion, opt, rnd):
        if (rnd + 1) < self.args.point:
        # if (rnd + 1) % self.args.point != 0:
            grad = torch.tensor([])
            layers = len(name_param)
            print("the number of layers is ", layers)
            index = 0
            for key in name_param.keys():
                if 'lora_A' in key:
                    lora_A = name_param[key]
                elif 'lora_B' in key:
                    lora_B = name_param[key]
                    if self.args.optimize == 1:
                        # llama
                        proporA = round(math.sqrt(self.args.proporA * self.args.proporB), 2)
                        proporB = round(self.args.proporA * self.args.proporB / proporA, 2)
                        self.args.proporA = proporA
                        self.args.proporB = proporB

                        if self.args.model == 'llama-2-7B' or self.args.model == 'llama-3.2-1B':
                            self.args.proporA = 0.2
                            self.args.proporB = 0.1
                        elif self.args.model == 'distilbert-base-multilingual-cased' or self.args.model == 'roberta-large':
                            self.args.proporA = 0.2
                            self.args.proporB = 0.1
                        self.args.optimize = 0
                        print("the optimization proportion are ", self.args.proporA, self.args.proporB)
                    # if self.args.optimize == 1 and opt and index == 0:
                    #     num_rowB = lora_B.shape[0]
                    #     num_colA = lora_A.shape[1]
                    #     rowB, indicesB = torch.sort(torch.norm(lora_B, p=2, dim=1) ** 2, descending=True)
                    #     colA, indicesA = torch.sort(torch.norm(lora_A, p=2, dim=0) ** 2, descending=True)
                    #     loss = float('inf')
                    #     target = 0
                    #     for proporA in np.arange(0.16, 0.35, 0.05):
                    #         proporB = 0.02 / proporA
                    #         index_B = int(proporB * num_rowB)
                    #         index_A = int(proporA * num_colA)
                            
                            
                    #         # unsel_B = torch.sum(rowB[index_B:])
                    #         # unsel_A = torch.sum(colA[index_A:])
                    #         # sel_B = torch.sum(rowB[:index_B])
                    #         # sel_A = torch.sum(colA[:index_A])
                    #         # if unsel_B * sel_A + unsel_A * sel_B + unsel_A * unsel_B < loss:
                    #         #     self.args.proporB = proporB
                    #         #     self.args.proporA = proporA
                    #         #     loss = unsel_B * sel_A + unsel_A * sel_B + unsel_A * unsel_B

                    #         tmp_target = 0
                    #         for i in range(index_B):
                    #             for j in range(index_A):
                    #                 tmp_target += torch.sum((lora_B[indicesB[i]] * lora_A[:, indicesA[j]])) ** 2
                    #                 # tmp_target += torch.norm(lora_B[indicesB[i]]) ** 2 + torch.norm(lora_A[:, indicesA[j]]) ** 2
                    #         if tmp_target > target:
                    #             self.args.proporB = proporB
                    #             self.args.proporA = proporA
                    #             target = tmp_target

                    #         # tmpB = lora_B[indicesB[:index_B]]
                    #         # tmpA = lora_A[:, indicesA[:index_A]]
                    #         # if torch.sum(torch.sum(tmpB, 0) * torch.sum(tmpA, 1)) > target:
                    #         #     target = torch.sum(torch.sum(tmpB, 0) * torch.sum(tmpA, 1))
                    #         #     self.args.proporB = proporB
                    #         #     self.args.proporA = proporA

                    #     print("the optimization proporA and prporB are ", self.args.proporA, self.args.proporB)
                    
                    num_rowB = int(self.args.proporB * lora_B.shape[0])
                    num_colA = int(self.args.proporA * lora_A.shape[1])
                    lora_B = self.topk_abs_rows(lora_B, num_rowB, rnd)
                    lora_A = self.topk_abs_columns(lora_A, num_colA, rnd)
                    result = (lora_B @ lora_A).view(-1)
                    print("rowB and colA are ", num_rowB, num_colA)
                    print('the shape and nnz of result are ', result.shape, torch.count_nonzero(result))
                    traffic = lora_B.numel() * proportion * (32 + 2 + int(math.log2(lora_B.numel())) + int(math.log2(layers))) + \
                                lora_A.numel() * proportion * (32 + 2 + int(math.log2(lora_A.numel())) + int(math.log2(layers)))
                    remain_traffic = traffic - num_rowB * int(math.log2(lora_B.shape[0]) + 1) - num_colA * int(math.log2(lora_A.shape[1]) + 1)
                    if self.args.quan == 0:
                        k = int(remain_traffic / (32 + 2 + int(math.log2(num_rowB * num_colA)) + int(math.log2(layers // 2))))
                    else:
                        k = int(remain_traffic / (2 + 2 + int(math.log2(num_rowB * num_colA)) + int(math.log2(layers // 2))))
                    # k = int(remain_traffic / (32 + 2 + int(math.log2(num_rowB * num_colA)) + int(math.log2(layers // 2))))
                    # k = int(proportion * result.numel())
                    print("the number of uploaded param is ", k)
                    print("the shape of loraA and loraB are ", lora_A.shape, lora_B.shape)
                    mask = torch.zeros_like(result, dtype=torch.bool)
                    v, i = torch.topk(result.abs(), k)
                    mask[i] = True
                    result[~mask] = 0
                    if self.args.quan != 0:
                        quan_v = PQ_quan(v * torch.sign(result[i]), 2 ** 2)
                        result[i] = quan_v
                    else:
                        result[i] = v * torch.sign(result[i])
                    grad = torch.cat((grad, result))
                    index += 1
                # else:
                #     l_grad = name_grad[key].view(-1)
                #     # k = int(proportion * l_grad.numel())
                #     # _, i = torch.topk(l_grad.abs(), k)
                #     # mask = torch.zeros(l_grad.numel())
                #     # mask[i] = 1
                #     # grad = torch.cat((grad, l_grad * mask))

                #     sigma = l_grad.std()
                #     k = int(proportion * l_grad.numel() * (32 + 2 + int(math.log2(l_grad.numel())) + int(math.log2(layers))) 
                #             / (1 + 2 + int(math.log2(l_grad.numel())) + int(math.log2(layers))))
                #     _, i = torch.topk(l_grad.abs(), k)
                #     mask = torch.zeros(l_grad.numel())
                #     mask[i] = 1
                #     grad = torch.cat((grad, torch.sign(l_grad * mask) * sigma))

        else:
            block = self.args.blocks
            self.d = self.trainable_parameters
            self.s = math.ceil(math.log(self.d, 2))
            self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            grad, block = self.cvlc_block_optim(grad=grad, param=param, blocks=block, rnd=rnd)
            # if self.args.quan == 1:
            #     block = self.args.blocks
            #     self.d = self.trainable_parameters
            #     self.s = math.ceil(math.log(self.d, 2))
            #     self.packet_min = int(self.packet_size / (self.max_bit + self.s))
            #     self.packet_max = int(self.packet_size / (self.min_bit + self.s))
            #     grad, block = self.cvlc_block_optim(grad=grad, param=param, blocks=block, rnd=rnd)
            # else:
            #     grad = self.compeft(proportion, name_grad, grad)
        return grad

    
    def compeft(self, proportion, name_grad, grad):
        k = int(proportion * grad.numel() * (32 + 1 + int(math.log2(grad.numel()))) 
                    / (1 + 1 + int(math.log2(grad.numel()))))
        _, i = torch.topk(grad.abs(), k)
        mask = torch.zeros(grad.numel())
        mask[i] = 1
        sigma = grad.std()
        return torch.sign(grad * mask) * sigma

        grad = torch.tensor([])
        layers = len(name_grad)
        for key in name_grad.keys():
            l_grad = name_grad[key].view(-1)
            sigma = l_grad.std()
            k = int(proportion * l_grad.numel() * (32 + 2 + int(math.log2(l_grad.numel())) + int(math.log2(layers))) 
                    / (1 + 2 + int(math.log2(l_grad.numel())) + int(math.log2(layers))))
            _, i = torch.topk(l_grad.abs(), k)
            mask = torch.zeros(l_grad.numel())
            mask[i] = 1
            grad = torch.cat((grad, torch.sign(l_grad * mask) * sigma))
        return grad
    
    def topk_AB(self, proportion, name_grad, grad):
        k = int(proportion * grad.numel())
        _, i = torch.topk(grad.abs(), k)
        mask = torch.zeros(grad.numel())
        mask[i] = 1
        return grad * mask

        grad = torch.tensor([])
        for key in name_grad.keys():
            l_grad = name_grad[key].view(-1)
            k = int(proportion * l_grad.numel())
            _, i = torch.topk(l_grad.abs(), k)
            mask = torch.zeros(l_grad.numel())
            mask[i] = 1
            grad = torch.cat((grad, l_grad * mask))
        return grad
    
    def topk_abs_rows(self, B, k, rnd):
        # 计算每一行元素绝对值的和
        # self.row_sums = torch.max(torch.abs(B), dim=1)[0]
        self.row_sums = torch.sum(B ** 2, dim=1)
        # 获取绝对值和最大的前 k 行的索引
        _, topk_indices = torch.topk(self.row_sums, k)
        # 创建一个布尔掩码，用于标记非 top-k 行
        mask = torch.ones(B.shape[0], dtype=torch.bool)
        mask[topk_indices] = False
        # 将非 top-k 行的元素置为 0
        B[mask] = 0
        return B
        
    def topk_abs_columns(self, A, k, rnd):
        # 计算矩阵 A 每列元素绝对值的和
        # self.col_sums = torch.max(torch.abs(A), dim=0)[0]
        self.col_sums = torch.sum(A ** 2, dim=0)
        # 找出绝对值和最大的前 k 列的索引
        _, topk_indices = torch.topk(self.col_sums, k)
        # 获取矩阵 A 的列数
        num_cols = A.size(1)
        # 遍历所有列
        for col in range(num_cols):
            if col not in topk_indices:
                # 如果该列不是前 k 大的列，将该列元素置为 0
                A[:, col] = 0
        return A

def randk(grad, k):
    """ Perform a topk operation on the gradient and return the processed gradient"""
    indices = torch.randperm(grad.numel())[:k]
    mask = torch.zeros_like(grad)
    mask[indices] = 1
    return grad * mask

def randk_AB(proportion, name_grad, name_param):
    grad = torch.tensor([])
    for key in name_grad.keys():
        l_grad = name_grad[key]
        k = int(proportion * l_grad.numel())
        i = torch.randperm(l_grad.numel())[:k]
        mask = torch.zeros(l_grad.numel())
        mask[i] = 1
        grad = torch.cat((grad, l_grad.view(-1) * mask))
    return grad

def remove_elements(tensor, indices):
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

# Update the index of topK
def new(proportion, name_grad, name_param):
    grad = torch.tensor([])
    index = 0
    for key in name_grad.keys():
        if index % 2 == 0:
            A_grad = name_grad[key].view(-1)
            A_param = name_param[key]
        else:
            B_grad = name_grad[key].view(-1)
            B_param = name_param[key]
            k = int(proportion * A_grad.numel())
            print("k is ", k)
            A_tmp = A_grad.clone().view(-1)
            B_tmp = B_grad.clone().view(-1)
            a = torch.norm(A_param, 2) ** 2 / 2
            b = torch.norm(B_param, 2) ** 2

            
            target = (B_param - B_grad.view(B_param.shape)) @ (A_param - A_grad.view(A_param.shape))
            selected_prop = [0.01 * i for i in range(101)]
            loss = float('inf')
            best_porpA, best_propB = 0., 0.
            for i in selected_prop:
                k_A = int(i * k)
                _, tmp_indicesA_1 = torch.topk(A_grad.abs(), k_A)
                _, tmp_indicesA_2 = torch.topk(-A_grad.abs(), k - k_A) 
                tmp_indicesA = torch.cat([tmp_indicesA_1, tmp_indicesA_2])
                select_A = A_grad[tmp_indicesA]
                for j in selected_prop:
                    k_B = int(j * k)
                    _, tmp_indicesB_1 = torch.topk(B_grad.abs(), k_B)
                    _, tmp_indicesB_2 = torch.topk(-B_grad.abs(), k - k_B) 
                    tmp_indicesB = torch.cat([tmp_indicesB_1, tmp_indicesB_2])
                    select_B = B_grad[tmp_indicesB]
                    # tmp_loss = 2 * torch.norm(select_A) ** 2 * torch.norm(select_B) ** 2 - a * torch.norm(select_B) ** 2 - b * torch.norm(select_A) ** 2
                    mask_A = torch.zeros(A_grad.shape)
                    mask_A[tmp_indicesA] = 1
                    mask_B = torch.zeros(B_grad.shape)
                    mask_B[tmp_indicesB] = 1
                    actual = (B_param - (B_grad * mask_B).view(B_param.shape)) @ (A_param - (A_grad * mask_A).view(A_param.shape))
                    tmp_loss = torch.norm(target - actual, 2) ** 2
                    if tmp_loss < loss:
                        loss = tmp_loss
                        indices_A = tmp_indicesA.clone()
                        indices_B = tmp_indicesB.clone()
                        best_porpA, best_propB = i, j


            # indices_A = []
            # indices_B = []
            # for _ in range(k):
            #     if 2 * torch.max(A_tmp.abs()) ** 2 - b <= 0:
            #         indices_A.append(torch.argmax(A_tmp.abs()))
            #         if 2 * A_tmp[indices_A[-1]] ** 2 - a >=0 :
            #             indices_B.append(torch.argmin(B_tmp.abs()))
            #         else:
            #             indices_B.append(torch.argmax(B_tmp.abs()))
            #         A_tmp = remove_elements(A_tmp, indices_A[-1])
            #         B_tmp = remove_elements(B_tmp, indices_B[-1])
            #         continue
            #     if 2 * torch.min(A_tmp.abs()) ** 2 - b >= 0:
            #         indices_A.append(torch.argmin(A_tmp.abs()))
            #         if 2 * A_tmp[indices_A[-1]] ** 2 - a >=0 :
            #             indices_B.append(torch.argmin(B_tmp.abs()))
            #         else:
            #             indices_B.append(torch.argmax(B_tmp.abs()))
            #         A_tmp = remove_elements(A_tmp, indices_A[-1])
            #         B_tmp = remove_elements(B_tmp, indices_B[-1])
            #         continue
            #     print("AAAAAAAA")
            #     # Compute matrix where matrix[i][j] = 2A[i]^2B[j]^2-aB[j]^2-bA[i]^2
            #     A_square = A_param.view(-1) ** 2
            #     B_square = B_param.view(-1) ** 2
            #     term1 = 2 * A_square.unsqueeze(1) * B_square
            #     term2 = b * A_square.unsqueeze(1)
            #     term3 = a * B_square
            #     matrix = term1 - term2 - term3
            #     _, min_index = torch.min(matrix.flatten(), dim=0)
            #     row, col = matrix.shape
            #     indices_A.append(min_index // col)
            #     indices_B.append(min_index % col)
                
            #     ## Compute i and j by default method
            #     # B_large = B_tmp[B_tmp >= math.sqrt(b / 2)]
            #     # B_small = B_tmp[B_tmp < math.sqrt(b / 2)]
            #     # large_A_index = torch.argmin(A_tmp.abs())
            #     # small_A_index = torch.argmax(A_tmp.abs())
            #     # if 2 * A_tmp[large_A_index] ** 2 - a >=0 :
            #     #     large_B_index = torch.argmin(B_large.abs())
            #     # else:
            #     #     large_B_index = torch.argmax(B_large.abs())
            #     # if 2 * A_tmp[small_A_index] ** 2 - a >=0 :
            #     #     small_B_index = torch.argmin(B_small.abs())
            #     # else:
            #     #     small_B_index = torch.argmax(B_small.abs())
            #     # large_loss = 2 * B_large[large_B_index] ** 2 * A_tmp[large_A_index] ** 2 - a * B_large[large_B_index] ** 2 - b * A_tmp[large_A_index] ** 2
            #     # small_loss = 2 * B_small[small_B_index] ** 2 * A_tmp[small_A_index] ** 2 - a * B_small[small_B_index] ** 2 - b * A_tmp[small_A_index] ** 2
            #     # if large_loss < small_loss:
            #     #     indices_A.append(large_A_index)
            #     #     indices_B.append(torch.where(B_tmp == B_large[large_B_index]))[0]
            #     # else:
            #     #     indices_A.append(small_A_index)
            #     #     indices_B.append(torch.where(B_tmp == B_small[small_B_index]))[0]
            print("len of indices A and B are ", len(indices_A), len(indices_B))
            print("the best proportion of A and B are", best_porpA, best_propB)
            mask = torch.zeros(A_grad.numel())
            mask[indices_A] = 1
            grad = torch.cat((grad, A_grad.view(-1) * mask))   
            mask = torch.zeros(B_grad.numel())
            mask[indices_B] = 1
            grad = torch.cat((grad, B_grad.view(-1) * mask))
        index += 1
    return grad

# Using SGD to update the value of topK
def testnew(proportion, name_grad, name_param):
    grad = torch.tensor([])
    index = 0
    for key in name_grad.keys():
        if index % 2 == 0:
            A_grad = name_grad[key]
            A_param = name_param[key]
        else:
            B_grad = name_grad[key]
            B_param = name_param[key]
            C = B_grad @ A_grad
            k = int(proportion * A_grad.numel())
            _, indices_A = torch.topk(A_grad.abs().view(-1), k)
            k = int(proportion * B_grad.numel())
            _, indices_B = torch.topk(B_grad.abs().view(-1), k)
            values_A = A_grad.flatten()[indices_A].clone().requires_grad_(True)
            values_B = B_grad.flatten()[indices_B].clone().requires_grad_(True)
            optimizer = optim.SGD([values_A, values_B], lr=0.5)
            num_epochs = 1500
            for epoch in range(num_epochs):
                sparse_A = torch.zeros(A_grad.numel())
                sparse_A = sparse_A.scatter(0, indices_A, values_A)
                sparse_A = sparse_A.view(A_grad.shape)
                sparse_B = torch.zeros(B_grad.numel())
                sparse_B = sparse_B.scatter(0, indices_B, values_B)
                sparse_B = sparse_B.view(B_grad.shape)
                sparse_C = sparse_B @ sparse_A
                loss = torch.norm(B_param @ (A_grad - sparse_A) + (B_grad - sparse_B) @ A_param - (C - sparse_C))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}, \
                          Spar: {torch.sum(sparse_A != 0) / sparse_A.numel()}, {torch.norm(sparse_C - C)}')
            sparse_A = torch.zeros(A_grad.numel())
            sparse_A[indices_A] = values_A
            sparse_B = torch.zeros(B_grad.numel())
            sparse_B[indices_B] = values_B
            grad = torch.cat((grad, sparse_A))
            grad = torch.cat((grad, sparse_B))
        index += 1
    return grad

def topk_rowcol(proportion, name_param):
    grad = torch.tensor([])
    for key in name_param.keys():
        param = name_param[key]
        result = torch.zeros_like(param)
        if 'lora_A' in key:
            v, i = torch.topk(param.abs(), dim=0, k = int(param.size(0) * proportion))
            cols = torch.arange(param.size(1))
            result[i.flatten(), cols] = v.flatten()
        elif 'lora_B' in key:
            v, i = torch.topk(param.abs(), dim=1, k = int(param.size(1) * proportion))
            row = torch.arange(param.size(0))
            result[row, i.flatten()] = v.flatten()
        grad = torch.cat((grad, result.view(-1)))
    return grad


def pq_pure(grad, bit_len):
    new_grad = PQ_quan(grad, 2 ** (bit_len))
    # new_grad = QSGD_quan(grad, 2 ** (bit_len - 1))
    return new_grad


# def fft(grad, thresholds=0.5):
#     fft_result = torch.fft.fft2(grad)
#     fft_shifted = torch.fft.fftshift(fft_result)
#     freq_center = (np.array(grad.shape) - 1) / 2
#     y, x = torch.meshgrid(torch.arange(grad.shape[0]), torch.arange(grad.shape[1]),
#                           indexing='ij')
#     distances = torch.sqrt((x - freq_center[1]) ** 2 + (y - freq_center[0]) ** 2)
#     radius = torch.max(distances) * thresholds
#     circular_filter = (distances >= radius).float()
#     filtered_fft = fft_shifted * circular_filter
#
#     # 将频谱进行反中心化
#     filtered_fft_unshifted = torch.fft.ifftshift(filtered_fft)
#
#     # 对过滤后的频谱进行逆傅里叶变换
#     filtered_parameters = torch.fft.ifft2(filtered_fft_unshifted).real
#     filtered_parameters = filtered_parameters.contiguous()
#     return filtered_parameters


def get_coefficient_of_variation(grad, blocks):
    len_split = len(grad)
    len_block = len_split // blocks
    list_block = [len_block] * blocks
    if sum(list_block) != len_split:
        list_block[-1] = list_block[-1] + len_split - sum(list_block)

    list_grad = list(torch.split(grad, list_block, dim=0))

    cov_list = []
    for l_grad in list_grad:
        grad = l_grad.abs()
        std_dev = torch.std(grad)
        mean = torch.mean(grad)
        cov = std_dev / mean
        cov_list.append(cov.item())
    return sum(cov_list) / len(cov_list)
