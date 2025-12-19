import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
matplotlib.rcParams['pdf.fonttype'] = 42

# plt.rcParams['font.family'] = 'SimHei'

path = 'gradient/llama-2-7B_'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
for i in range(1, 2, 1):
    file = path + str(i) + '.pt'
    name_grad = torch.load(file)
    idx = 0
    for k, v in name_grad.items():
        if idx == 0:
            lora_A = v
            result = torch.norm(lora_A, dim=0) ** 2
            # # result = lora_B @ lora_A
            result, _ = torch.sort(result.view(-1).abs(), descending=True)
            x = range(1, result.numel()+1)
            ax1.plot(x, result, label='real')

            #  拟合幂律分布
            def power(x, alpha, c):
                # return c * np.log(alpha*x)
                # return c * np.exp(-alpha*x)
                return c * np.power(x, alpha)
            def log(x, alpha, d):
                return -0.01*np.log(alpha*x) + d
            def exp(x, alpha):
                return 0.001*np.exp(-alpha*x)
            
            x = torch.tensor(range(1, result.numel() + 1))
            powert, _ = curve_fit(power, x.numpy(), result.numpy())
            ax1.plot(x, powert[1]*np.power(x, powert[0]), label='fitted by power law', linestyle='--', marker='^', markevery=50)
            logt, _ = curve_fit(log, x.numpy(), result.numpy())
            ax1.plot(x, -0.01*np.log(x * logt[0]) + logt[1], label='fitted by log', linestyle='--', marker='.', markevery=50)
            expt, _ = curve_fit(exp, x.numpy(), result.numpy())
            ax1.plot(x, 0.001*np.exp(-x * expt[0]), label='fitted by exp', linestyle='--', marker='*', markevery=50)
            ax1.legend(loc='best')
            ax1.set_xlabel('rank ID', size=12)
            ax1.set_ylabel('value', size=12)
            ax1.set_title("low-rank matrix $\mathbf{A}$", y=-0.3)
        else:
            lora_B = v
            result = torch.norm(lora_B, dim = 1) ** 2
            # result = lora_B @ lora_A
            result, _ = torch.sort(result.view(-1).abs(), descending=True)
            x = range(1, result.numel()+1)
            ax2.plot(x, result, label='real')

            #  拟合幂律分布
            def power(x, alpha, c):
                # return c * np.log(alpha*x)
                # return c * np.exp(-alpha*x)
                return c * np.power(x, alpha)
            def log(x, alpha, d):
                return -0.0001*np.log(alpha*x) + d
            def exp(x, alpha):
                return 0.00001*np.exp(-alpha*x)
            
            x = torch.tensor(range(1, result.numel() + 1))
            powert, _ = curve_fit(power, x.numpy(), result.numpy())
            ax2.plot(x, powert[1]*np.power(x, powert[0]), label='fitted by power law', linestyle='--', marker='^', markevery=50)
            logt, _ = curve_fit(log, x.numpy(), result.numpy())
            ax2.plot(x, -0.0001*np.log(x * logt[0]) + logt[1], label='fitted by log', linestyle='--', marker='.', markevery=50)
            expt, _ = curve_fit(exp, x.numpy(), result.numpy())
            ax2.plot(x, 0.00001*np.exp(-x * expt[0]), label='fitted by exp', linestyle='--', marker='*', markevery=50)
            ax2.legend(loc='best')
            ax2.set_xlabel('rank ID', size=12)
            ax2.set_ylabel('value', size=12)
            ax2.set_title("low-rank matrix $\mathbf{B}$", y=-0.3)
            break
        idx += 1

plt.show()