import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.use('TkAgg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
        }


# fig = plt.figure(figsize=(5.7, 3.7))
fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
models = ['roberta_20news_iid', '3llama_20news_iid', 'llama_20news_iid']
titles = ['RoBERTa(0.3B)', 'LLaMA 3.2(1B)', 'LLaMA 2(7B)']
traffic = [140 * 1500 / 1024 / 1024, 140 * 1500 / 1024 / 1024, 152 * 1500 / 1024 / 1024, 152 * 1500 / 1024 / 1024,
           782 * 1500 / 1024 / 1024, 782 * 1500 / 1024 / 1024]

# model = 'distilbert_20news_iid'
# model = 'largeroberta_20news_iid'
# model = '3llama_20news_iid'

for i in range(3):
    model = models[i]
    compeft = []
    with open('accu/' + model + '_compeft.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            compeft.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    prune = []
    with open('accu/' + model + '_prune.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            prune.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    topk = []
    with open('accu/' + model + '_topk_AB.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            topk.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    updateW = []
    with open('accu/' + model + '_updateW.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            updateW.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    optim = []
    with open('accu/' + model + '_optim.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            optim.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    block = []
    with open('accu/' + model + '_block.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            block.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    inter = 1
    x = range(1, 201)
    compeft = compeft[:len(x)]
    topk = topk[:len(x)]
    prune = prune[:len(x)]
    updateW = updateW[:len(x)]
    optim = optim[:len(x)]
    block = block[:len(x)]

    axs[i].set_xlabel("Global Iterations", size=15, family='Times New Roman')
    axs[i].set_ylabel("Accuracy", size=15, family='Times New Roman')
    if i <= 0:
        markevery = 5
    else:
        markevery = 1
    axs[i].plot(x[:len(optim)], optim, marker='x', markersize=5, label="FLR-SS", markevery=markevery, markerfacecolor='white')
    axs[i].plot(x[:len(updateW)], updateW, marker='*', markersize=5, label="FLR-SS(w/o)", markevery=markevery, markerfacecolor='white')
    axs[i].plot(x[:len(compeft)], compeft, marker='D', markersize=5, label="ComPEFT", markevery=markevery, markerfacecolor='white')
    axs[i].plot(x[:len(prune)], prune, marker='^', markersize=5, label="Sparseadapter", markevery=markevery, markerfacecolor='white')
    axs[i].plot(x[:len(topk)], topk, marker='+', markersize=5, label="FLASC", markevery=markevery, markerfacecolor='white')
    axs[i].plot(x[:len(block)], block, marker='.', markersize=5, label="FLM-TopK", markevery=markevery, markerfacecolor='white')
    # plt.title('roberta-large (0.3B)')
    axs[i].set_ylim(0.3,)
    if i == 0:
        axs[i].legend(bbox_to_anchor=(0, 1.07), loc=6, ncol=6, borderaxespad=0, prop=font1)
    axs[i].set_title(titles[i], y=-0.3)

    print("Iteration: {}, Optim: {}, Block: {}, Improvement: {}".
          format(i, max(optim), max(block), (max(optim)-max(block)) * 100))

    
    # target_accu = int(max(compeft) * 100)
    # updateW_tra = 10 * (np.where(np.array(updateW) >= target_accu / 100)[0][0] + 1) * traffic[i]
    # optim_tra = 10 * (np.where(np.array(optim) >= target_accu / 100)[0][0] + 1) * traffic[i]
    # block_tra = 10 * (np.where(np.array(block) >= target_accu / 100)[0][0] + 1) * traffic[i]
    # topk_tra = 10 * (np.where(np.array(topk) >= target_accu / 100)[0][0] + 1) * traffic[i]
    # compeft_tra = 10 * (np.where(np.array(compeft) >= target_accu / 100)[0][0] + 1) * traffic[i]
    # print("Target: {}, Iteration: {}, optim: {}, UpdateW: {}, block: {}, topk: {}, Compeft: {}, Reduce: {}".
    #       format(target_accu, i, optim_tra, updateW_tra, block_tra, topk_tra, compeft_tra, 
    #              (block_tra-optim_tra) / block_tra), )
    # print()
plt.show()
