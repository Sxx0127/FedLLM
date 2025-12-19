import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
        }


# fig = plt.figure(figsize=(5.7, 3.7))
fig, axs = plt.subplots(1, 6, figsize=(8, 3))
model = ['distilbert_20news_iid', 'distilbert_20news_niid', '3llama_20news_iid', '3llama_20news_niid', 
         'llama_20news_iid', 'llama_20news_niid']

# model = 'distilbert_20news_iid'
# model = 'largeroberta_20news_iid'
# model = '3llama_20news_iid'

for i in range(6):

    # raw = []
    # with open('accu/' + model + '_raw.txt', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for idx, line in enumerate(lines):
    #         raw.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
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
    # prune = prune[4:]
    topk = []
    with open('accu/' + model + '_topk_AB.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            topk.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
    # STopK = []
    # with open('accu/' + model + '_STopK.txt', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for idx, line in enumerate(lines):
    #         STopK.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))
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

    # quanUp = []
    # with open('accu/' + model + '_quanUp.txt', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for idx, line in enumerate(lines):
    #         quanUp.append(float(line[line.find('acc') + 6:line.find('eval_runtime') - 3]))



    inter = 1
    x = range(1, 201)
    # raw = raw[:len(x)]
    compeft = compeft[:len(x)]
    topk = topk[:len(x)]
    prune = prune[:len(x)]
    # STopK = STopK[:len(x)]
    updateW = updateW[:len(x)]
    optim = optim[:len(x)]
    block = block[:len(x)]
    # quanUp = quanUp[:len(x)]
    plt.xlabel("Global Round", size=15, family='Times New Roman')
    plt.ylabel("Accuracy", size=15, family='Times New Roman')
    plt.plot(x[:len(optim)], optim, marker='x', markersize=5, label="optim", markevery=10, markerfacecolor='white')
    plt.plot(x[:len(updateW)], updateW, marker='*', markersize=5, label="new", markevery=1, markerfacecolor='white')
    # plt.plot(x[:len(quanUp)], quanUp, marker='+', markersize=5, label="new+quan(2bit)", markevery=10, markerfacecolor='white')
    plt.plot(x[:len(compeft)], compeft, marker='D', markersize=5, label="compeft", markevery=1, markerfacecolor='white')
    plt.plot(x[:len(prune)], prune, marker='^', markersize=5, label="Sparseadapter", markevery=1, markerfacecolor='white')
    # plt.plot(x[:len(raw)], raw, marker='*', markersize=5, label="Uncompressed", markevery=10, markerfacecolor='white')
    plt.plot(x[:len(topk)], topk, marker='+', markersize=5, label="topk", markevery=1, markerfacecolor='white')
    plt.plot(x[:len(block)], block, marker='+', markersize=5, label="FLM-TopK", markevery=1, markerfacecolor='white')
    # plt.title('roberta-large (0.3B)')
    plt.ylim(0.35,)
    plt.legend(bbox_to_anchor=(0, 1.05), loc=3, ncol=3, borderaxespad=0, prop=font1)
plt.show()
