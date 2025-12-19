import copy

import numpy as np


def get_partition_map(partition: bool, label: bool, tokenized_datasets, num_client: int, alpha: float, label_list: list,
                      num_labels: int):
    if partition:
        partition_map = get_iid_partition(tokenized_datasets=tokenized_datasets, num_client=num_client)
    elif label:
        partition_map = get_noniid_label_partition(tokenized_datasets=tokenized_datasets, num_client=num_client,
                                                   alpha=alpha,
                                                   num_label=num_labels, label_list=label_list)
    else:
        partition_map = get_noniid_num_partition(tokenized_datasets=tokenized_datasets, num_client=num_client,
                                                 alpha=alpha)
    return partition_map


def get_iid_partition(tokenized_datasets, num_client: int):
    num_samples = len(tokenized_datasets['train'])
    samples_per_client = (np.repeat(num_samples, num_client) / num_client).astype(int)

    all_train_indices = np.arange(num_samples)
    np.random.shuffle(all_train_indices)
    indices_split = np.split(all_train_indices, np.cumsum(samples_per_client)[:-1])
    partition_map = {i: indices_split[i].tolist() for i in range(num_client)}
    return partition_map


def get_noniid_num_partition(tokenized_datasets, num_client: int, alpha: float):
    num_samples = len(tokenized_datasets['train'])
    proportions = np.random.dirichlet(np.repeat(alpha, num_client))
    samples_per_client = (proportions * num_samples).astype(int)
    zero_indices = np.argwhere(samples_per_client == 0).flatten()
    samples_per_client[zero_indices] += 1

    samples_per_client[-1] = num_samples - samples_per_client[:-1].sum()

    if samples_per_client[-1] <= 0:
        k = samples_per_client[-1]
        samples_per_client[-1] = 1
        max_element = np.max(samples_per_client)
        max_index = np.argmax(samples_per_client)
        max_element = max_element + k - 1
        samples_per_client[max_index] = max_element

    all_train_indices = np.arange(num_samples)
    np.random.shuffle(all_train_indices)
    indices_split = np.split(all_train_indices, np.cumsum(samples_per_client)[:-1])
    partition_map = {i: indices_split[i].tolist() for i in range(num_client)}
    return partition_map


def get_noniid_label_partition(tokenized_datasets, num_client: int, alpha: float, num_label: int, label_list: list):
    num_samples = len(tokenized_datasets['train'])
    label_distribution = np.random.dirichlet([alpha] * num_label, num_client)
    proportions = np.array([1 / num_client] * num_client)
    samples_per_client = (proportions * num_samples).astype(int)
    samples_per_client[-1] = num_samples - samples_per_client[:-1].sum()
    copy_samples_per_client = samples_per_client
    indices_split = [[] for _ in range(num_client)]
    label_distribution_cumsum = np.cumsum(label_distribution, axis=1)
    label_data = []
    for label in range(num_label):
        arr = np.array(label_list)
        indices = np.where(arr == label)[0]
        label_data.append(indices)
    cls_amount = [len(label_data[i]) for i in list(range(len(label_data)))]
    copy_cls_amount = copy.deepcopy(cls_amount)
    while np.sum(samples_per_client) != 0:
        curr_clnt = np.argmax(samples_per_client)
        if copy_samples_per_client[curr_clnt] <= 0:
            continue
        copy_samples_per_client[curr_clnt] -= 1
        curr_prior = label_distribution_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            if cls_amount[cls_label] <= 0:
                if alpha < 1:
                    cls_amount[cls_label] = copy_cls_amount[cls_label]
                continue
            cls_amount[cls_label] -= 1
            indices_split[curr_clnt].append(label_data[cls_label][cls_amount[cls_label]])
            break
    partition_map = {i: indices_split[i] for i in range(num_client)}
    return partition_map
