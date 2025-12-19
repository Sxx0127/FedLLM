import os
import torch
import random
import warnings
import argparse
import numpy as np
import transformers

from fed_trainer import Fed_trainer



def args_parser():
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='20_newsgroups', help='dataset used for training'
                                                                             '(20_newsgroups)')
    parser.add_argument('--subdataset', type=str, default='cola', help='subdataset name for dataset')

    parser.add_argument('--partition', action='store_true', help='iid:True, non-iid:False')
    parser.add_argument('--label', action='store_true', help='using label to partition')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha for the dirichlet distribution for data partitioning')

    # Federated Learning configuration
    parser.add_argument('--centralized', action='store_true', help='using centralized learning')
    parser.add_argument('--sample_fraction', type=float, default=0.1,
                        help='how many clients are sampled in each round')
    parser.add_argument('--num_client', type=int, default=100, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=150, help='number of maximum communication round')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--test_bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--save_model', action='store_true', help='save model after training')
    parser.add_argument('--save_metric', action='store_true', help='save metric after training')

    # Model configuration
    parser.add_argument('--model', type=str, default='roberta-large',
                        help='models distilbert-base-cased distilbert-base-multilingual-cased roberta-large llama-2-7B roberta-base llama-3.2-1B')
    parser.add_argument('--max_length', type=int, default=256, help='max_length for data')

    # LoRa configuration
    parser.add_argument('--lora_alpha', type=int, default=32, help='scaling factors for LoRa')
    parser.add_argument('--lora_rank', type=int, default=16, help='rank of matrices in LoRa')

    # Compress configuration
    parser.add_argument('--packet_num', type=int, default=360, help='number of packet in each communication round')
    parser.add_argument('--method', default='raw', type=str,
                        help='methods to compress, '
                             'including raw, cvlc, fft, topk, layer, matrix, block, block_opt, topk_block')
    parser.add_argument('--topk_method', type=str, default='gradient', help='how to find k for top_k')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping if accuracy exceeds 60%')
    parser.add_argument('--blocks', type=int, default=100, help='the number of block')
    parser.add_argument('--bit', type=int, default=10, help='the number of the length of bits in topk_pq')
    parser.add_argument('--residual', action='store_true', help="use residual model")
    
    # parser.add_argument('--proporA', type=float, default=0.2, help='')
    # parser.add_argument('--proporB', type=float, default=0.1, help='')
    parser.add_argument('--proporA', type=float, default=0.4, help='')
    parser.add_argument('--proporB', type=float, default=0.05, help='')
    parser.add_argument('--proportion', type=float, default=0.02, help='')
    parser.add_argument('--optimize', type=int, default=0, help='')
    parser.add_argument('--point', type=int, default=2, help='')
    parser.add_argument('--quan', type=int, default=0, help='')
    parser.add_argument('--factor', type=float, default=1, help='')
    parser.add_argument('--weight', type=float, default=0.02, help='')

    # Computation configuration
    # parser.add_argument('--device', type=str, default='2', help='the device to run the program')
    parser.add_argument('--random_seed', type=int, default=62, help="random seed")
    parser.add_argument('--GPU', type=int, default=0, help="random seed")

    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arguments.GPU)  #（代表仅使用第0，1号GPU）
    return arguments


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)


def run(arguments):
    random_list = [random.randint(1, 100) for _ in range(arguments.comm_round)]
    trainer = Fed_trainer(args=arguments, random_list=random_list)
    trainer.run()
    return


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    args = args_parser()
    set_random_seed(args.random_seed)
    if args.centralized:
        args.num_client = 1
        args.sample_fraction = 1
    if args.method == 'fedcomp':
        args.residual = True
    run(args)
