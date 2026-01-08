import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
matplotlib.rcParams['pdf.fonttype'] = 42

# plt.rcParams['font.family'] = 'SimHei'

def topk_keep_global(matrix, k, largest=True):
    flat_mat = matrix.flatten()
    _, topk_flat_indices = torch.topk(flat_mat.abs(), k, largest=largest)
    
    flat_mask = torch.zeros_like(flat_mat)
    flat_mask[topk_flat_indices] = 1
    mask = flat_mask.reshape(matrix.shape)
    
    result = matrix * mask
    return result

def topk_abs_rows(B, k):
    row_sums = torch.sum(B ** 2, dim=1)
    _, topk_indices = torch.topk(row_sums, k)
    mask = torch.ones(B.shape[0], dtype=torch.bool)
    mask[topk_indices] = False
    B[mask] = 0
    return B
    
def topk_abs_columns(A, k):
    col_sums = torch.sum(A ** 2, dim=0)
    _, topk_indices = torch.topk(col_sums, k)
    num_cols = A.size(1)
    for col in range(num_cols):
        if col not in topk_indices:
            A[:, col] = 0 
    return A

models = ['distilbert-base-multilingual-cased_11.pt', 'roberta-large_10.pt', 'llama-2-7B_5.pt']
paths = ['distilbert-base-multilingual-cased_', 'roberta-large_']
markevery = 15
fig, axs = plt.subplots(1, 2, figsize=(7, 3))
for j, path in enumerate(paths):
    error1s = []
    error2s = []
    iteration = 11
    # for i, model in enumerate(models):
    for i in range(1, iteration):
        model = path + str(i) + '.pt'
        name_grad = torch.load(model)
        idx = 0
        target_diff = -100
        target_error1 = 0
        target_error2 = 0
        target_idx = 0
        for k, v in name_grad.items():
            if idx % 2 == 0:
                lora_A = v
            else:
                lora_B = v
                accu_re = lora_B @ lora_A

                spar_rate = 0.02
                num_A = int(lora_A.numel() * spar_rate)
                num_B = int(lora_B.numel() * spar_rate)
                traffic = num_A * (32 + math.log2(lora_A.numel())) + num_B * (32 + math.log2(lora_A.numel()))

                spar_A = topk_keep_global(lora_A, num_A)
                spar_B = topk_keep_global(lora_B, num_B)
                result1 = spar_B @ spar_A
                error1 = (torch.norm(accu_re - result1) ** 2) / (torch.norm(accu_re) ** 2)
                print("compression error1 is ", error1)

                proporA = 0.1
                proporB = 0.1
                col_A = topk_abs_columns(lora_A, int(proporA * lora_A.shape[1]))
                row_B = topk_abs_rows(lora_B, int(proporB * lora_B.shape[0]))
                spar_re = row_B @ col_A
                remain_traffic = traffic - int(proporA * lora_A.shape[1]) * math.log2(lora_A.shape[1]) - int(proporB * lora_B.shape[1]) * math.log2(lora_B.shape[0])
                k = int(remain_traffic / (32 + math.log2(spar_re.numel() * proporA * proporB)))
                # print("k is ", k)
                # k = int(traffic / (32 + math.log2(spar_re.numel())))
                # print("k is ", k)
                result2 = topk_keep_global(spar_re, k)
                error2 = (torch.norm(accu_re - result2) ** 2) / (torch.norm(accu_re) ** 2)
                print("compression error2 is ", error2)
                if error1 - error2 > target_diff:
                    target_diff = error1 - error2
                    target_error1 = error1
                    target_error2 = error2
                    target_idx = idx - 1
            idx += 1
        # print('target diff is ', target_diff)
        error1s.append(target_error1)
        error2s.append(target_error2)
    print(target_diff, target_idx)
    axs[j].plot(range(1, iteration), error1s, label='origin error', linestyle='--', marker='^')
    axs[j].plot(range(1, iteration), error2s, label='optim error', linestyle='--', marker='*')
    # axs.legend(loc='best')
    axs[j].legend(bbox_to_anchor=(0, 1.07), loc=6, ncol=6, borderaxespad=0)
    axs[j].set_xlabel('Iteration', size=12)
    axs[j].set_ylabel('Compression Error', size=12)
plt.show()
