import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
matplotlib.rcParams['pdf.fonttype'] = 42

# plt.rcParams['font.family'] = 'SimHei'

models = ['distilbert-base-multilingual-cased_1.pt', 'roberta-large_1.pt']
markevery = 15
fig, axs = plt.subplots(1, 2, figsize=(7, 3))
for i, model in enumerate(models):
    name_grad = torch.load(model)
    idx = 0
    for k, v in name_grad.items():
        if idx == 1:
            print(k)
            lora = v
            result = torch.norm(lora, dim=idx) ** 2
            # # result = lora_B @ lora_A
            result, _ = torch.sort(result.view(-1).abs(), descending=True)
            x = range(1, result.numel()+1)
            axs[i].plot(x, result, label='real', markevery=int(result.numel() / markevery))

            #  拟合幂律分布
            def power(x, alpha, c):
                # return c * np.log(alpha*x)
                # return c * np.exp(-alpha*x)
                return c * np.power(x, alpha)
            def log(x, alpha, d):
                return -max(result) / 2 * np.log(alpha*x) + d
            def exp(x, alpha):
                return max(result) / 10 * np.exp(-alpha*x)
            
            x = torch.tensor(range(1, result.numel() + 1))
            powert, _ = curve_fit(power, x.numpy(), result.numpy())
            axs[i].plot(x, power(x, powert[0], powert[1]), label='fitted by power law', linestyle='--', marker='^', markevery=int(result.numel() / markevery))
            logt, _ = curve_fit(log, x.numpy(), result.numpy())
            axs[i].plot(x, log(x, logt[0], logt[1]), label='fitted by log', linestyle='--', marker='.', markevery=int(result.numel() / markevery))
            expt, _ = curve_fit(exp, x.numpy(), result.numpy())
            axs[i].plot(x, exp(x, expt[0]), label='fitted by exp', linestyle='--', marker='*', markevery=int(result.numel() / markevery))
            # axs[i].legend(loc='best')
            axs[i].set_xlabel('rank ID', size=12)
            axs[i].set_ylabel('value', size=12)
            if i == 0:
                axs[i].legend(bbox_to_anchor=(0, 1.07), loc=6, ncol=6, borderaxespad=0)
            # axs[i].set_title("low-rank matrix $\mathbf{A}$", y=-0.3)
            break
        # else:
        #     lora_B = v
        #     result = torch.norm(lora_B, dim = 1) ** 2
        #     # result = lora_B @ lora_A
        #     result, _ = torch.sort(result.view(-1).abs(), descending=True)
        #     x = range(1, result.numel()+1)
        #     axs[i].plot(x, result, label='real')

        #     #  拟合幂律分布
        #     def power(x, alpha, c):
        #         # return c * np.log(alpha*x)
        #         # return c * np.exp(-alpha*x)
        #         return c * np.power(x, alpha)
        #     def log(x, alpha, d):
        #         return -0.0001*np.log(alpha*x) + d
        #     def exp(x, alpha):
        #         return 0.00001*np.exp(-alpha*x)
            
        #     x = torch.tensor(range(1, result.numel() + 1))
        #     powert, _ = curve_fit(power, x.numpy(), result.numpy())
        #     axs[i].plot(x, powert[1]*np.power(x, powert[0]), label='fitted by power law', linestyle='--', marker='^', markevery=50)
        #     logt, _ = curve_fit(log, x.numpy(), result.numpy())
        #     axs[i].plot(x, -0.0001*np.log(x * logt[0]) + logt[1], label='fitted by log', linestyle='--', marker='.', markevery=50)
        #     expt, _ = curve_fit(exp, x.numpy(), result.numpy())
        #     axs[i].plot(x, 0.00001*np.exp(-x * expt[0]), label='fitted by exp', linestyle='--', marker='*', markevery=50)
        #     axs[i].legend(loc='best')
        #     axs[i].set_xlabel('rank ID', size=12)
        #     axs[i].set_ylabel('value', size=12)
        #     axs[i].set_title("low-rank matrix $\mathbf{B}$", y=-0.3)
        #     break
        idx += 1

plt.show()
