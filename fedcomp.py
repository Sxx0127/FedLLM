import copy
import math
import torch
import numpy as np

from cvlc import pq_pure


def split_tensor_to_vectors(tensor, index):
    new_tensor = torch.zeros_like(tensor)
    vectors = [tensor[i, :].unsqueeze(0) for i in range(tensor.shape[0])]
    for i in range(tensor.shape[0]):
        new_tensor[i, :] = index
        index += 1
    return vectors, new_tensor, index


def get_param(gobal_model, model):
    grad = {}
    param = {}
    for layer in model.state_dict():
        if layer.find("num_batches_tracked") != -1:
            continue
        if 'lora' in layer:
            param_now = model.state_dict()[layer].detach().cpu()
            param_last = gobal_model.state_dict()[layer].detach().cpu()
            param_g = param_last - param_now
            param[layer] = param_now
            grad[layer] = param_g
    return grad, param


class Fedcomp(object):
    def __init__(self, args):
        super().__init__()
        self.k = 0
        self.args = args
        self.lora_rank = args.lora_rank
        self.total_size = 1500 * 8 * args.packet_num
        self.vector_len = 4096

    def get_l1_norm(self, tensor: torch.tensor):
        l1_norm = torch.mean(torch.abs(tensor))
        if tensor.size(1) == self.vector_len:
            l1_norm *= 1.0
        return l1_norm

    def get_top_k_vector(self, vector_list: list[torch.tensor]):
        l1_list = []
        for idx, i in enumerate(vector_list):
            l1 = self.get_l1_norm(i)
            if i.size(1) == self.lora_rank:
                index_len = self.vector_len
            else:
                index_len = self.lora_rank
            l1_list.append((idx, l1.item(), 32 * i.size(1) + 5 + math.ceil(math.log(index_len, 2))))
        l1_list.sort(key=lambda x: x[1], reverse=True)
        x2 = list(map(lambda x: x[2], l1_list))
        k = calculate_n(x2, self.total_size)
        max_indices = list(map(lambda x: x[0], l1_list))[:k]
        return max_indices

    def fed_comp(self, gobal_model, model):
        grad, param = get_param(gobal_model, model)
        all_vectors = []
        mark_dict = {}
        index = 0

        for key, tensor in grad.items():
            vectors, new_tensor, index = split_tensor_to_vectors(tensor, index)
            mark_dict[key] = new_tensor
            all_vectors.extend(vectors)

        new_grad = []
        res_grad = []
        max_indices = self.get_top_k_vector(all_vectors)
        for key, tensor in grad.items():
            mark = mark_dict[key].numpy()
            numpy_matrix = tensor.numpy()
            numpy_matrix_res = copy.deepcopy(numpy_matrix)

            mask = np.isin(mark, max_indices)
            numpy_matrix[~mask] = 0
            numpy_matrix_res[mask] = 0
            # numpy_matrix_res = numpy_matrix_res * 0.0

            result_tensor = torch.from_numpy(numpy_matrix)
            result_tensor_res = torch.from_numpy(numpy_matrix_res)

            new_grad.append(result_tensor.view(-1))
            res_grad.append(result_tensor_res.view(-1))
        res = torch.cat(res_grad)
        new_res = pq_pure(res, 4)
        return torch.cat(new_grad), new_res


def top_k_indices(nums, k):
    # 获取元素及其下标，并按元素值排序
    indexed_nums = list(enumerate(nums))
    indexed_nums.sort(key=lambda x: x[1], reverse=True)

    # 提取前 k 个元素的下标
    top_k = [idx for idx, _ in indexed_nums[:k]]

    return top_k


def calculate_n(arr, k):
    n = 0  # 初始化项数为0
    current_sum = 0  # 初始化当前累加和为0

    for i in range(len(arr)):
        current_sum += arr[i]  # 累加当前元素

        if current_sum >= k:
            break  # 如果当前累加和超过k，则退出循环

        n += 1  # 增加项数

    return n
