import torch

def QSGD_loss(x, xb, parameter):
        # return min(x / (2 ** (2 * xb)), math.sqrt(x) / (2 ** xb)) * torch.sum(parameter)
        return x / (2 ** (2 * xb)) * torch.sum(parameter)

def PQ_loss(x, xb, parameter):
    return x / (2 * ((2 ** xb - 1) ** 2)) * torch.sum(parameter)


def PQ_quan(parameters, centroids):
    # ans = torch.zeros_like(parameters)
    # abs_para = parameters.abs()
    # p_max = torch.max(abs_para)
    # p_min = torch.min(abs_para)
    # interval = (p_max - p_min) / centroids
    # left = ((abs_para - p_min) / interval).int() * interval + p_min
    # right = left + interval
    # probability = (abs_para - left) / (right - left)
    # probability[probability < 1e-5] = 0
    # seed = torch.rand(parameters.shape)
    # ans[seed < probability] = right[seed < probability]
    # ans[seed >= probability] = left[seed >= probability]
    # return ans * torch.sign(parameters)
    ans = torch.zeros_like(parameters)
    p_max = torch.max(parameters)
    p_min = torch.min(parameters)
    interval = (p_max - p_min) / (centroids - 1)
    left = ((parameters - p_min) / interval).int() * interval + p_min
    right = left + interval
    probability = (parameters - left) / (right - left)
    probability[probability < 1e-5] = 0
    seed = torch.rand(parameters.shape)
    ans[seed < probability] = right[seed < probability]
    ans[seed >= probability] = left[seed >= probability]
    return ans

def QSGD_quan(parameters, centroids):
    norm = torch.norm(parameters, p=2)
    rang = parameters.abs() / norm
    l = (rang * centroids).int()
    p = 1 - (rang * centroids - l)
    p[p < 1e-5] = 0
    seed = torch.rand(parameters.shape)
    xi = torch.zeros_like(parameters)
    xi[seed < p] = l[seed < p] / centroids
    xi[seed >= p] = (l + 1)[seed >= p] / centroids
    ans = norm * torch.sign(parameters) * xi
    return ans