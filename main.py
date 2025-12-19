import argparse
# from xglue_utils import load_xglue
import os

# Custom Library
from utils import set_random_seed, load_and_prepare_20newsgroups, partition_xglue, \
    load_and_prepare_20newsgroups_non_iid_label


# os.environ["http_proxy"] = "http://127.0.0.1:20171"
# os.environ["https_proxy"] = "http://127.0.0.1:20171"n // 和 / 的区别


# os.environ["http_proxy"] = "http://172.31.72.197:20171"
# os.environ["https_proxy"] = "http://172.31.72.197:20171"

# --dataset 20_newsgroups --n_parties 20
def run(args):
    print(args)
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    # Data partitioning based on non-iid strategy
    print('* Partitioning data (num_party: {} by {}, alpha: {})'.format(args.n_parties, args.partition, args.alpha))
    if args.dataset == "20_newsgroups":
        if args.label:
            print("Splitting data using labels")
            client2data, _ = load_and_prepare_20newsgroups_non_iid_label(model=args.model,
                                                                         n_parties=args.n_parties,
                                                                         alpha=args.alpha)
        else:
            client2data, _ = load_and_prepare_20newsgroups(dataset=args.dataset,
                                                           model=args.model,
                                                           datadir=args.datadir,
                                                           partition=args.partition,
                                                           n_parties=args.n_parties,
                                                           alpha=args.alpha)

    elif args.dataset == "xglue":
        client2data = partition_xglue(model=args.model, subdataset=args.subdataset, n_parties=args.n_parties,
                                      partition=args.partition, alpha=args.alpha)
    else:
        exit("No implementation error")
    # Select Solver based on learning strategy
    solver = None
    if args.alg == 'full':
        from solvers.full_fed_avg import FullFedAvgSolver
        solver = FullFedAvgSolver(args=args, client2data=client2data)
    elif args.alg == 'new':
        from solvers.lora_fed_avg import HyperFedAvgSolver
        solver = HyperFedAvgSolver(args=args, client2data=client2data)
    solver.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='20_newsgroups', help='dataset used for training')
    parser.add_argument('--subdataset', type=str, default='nc', help='subdataset name for xglue')
    #
    parser.add_argument('--partition', type=bool, default=False, help='non-iid:True, iid:False')
    parser.add_argument('--label', action='store_true', help='using label to partition')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha for the dirichlet distribution for data partitioning')

    # Federated Learning configuration
    parser.add_argument('--sample_fraction', type=float, default=0.25,
                        help='how many clients are sampled in each round')
    parser.add_argument('--n_parties', type=int, default=20, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=10, help='number of maximum communication round')

    # Training configuration
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64),'
                                                                   'if using llama-2-7B, recommend batch_size=8')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')

    # Model configuration
    parser.add_argument('--alg', type=str, default='new', help='communication strategy: full/last')
    # distilbert-base-multilingual-cased, roberta-base, llama-2-7B
    parser.add_argument('--model', type=str, default='distilbert-base-multilingual-cased', help='models')

    parser.add_argument('--rank', type=int, default=16, help='dimension of bottleneck layers')
    parser.add_argument('--con_dim', type=int, default=32, help='dimension of latent factors')
    parser.add_argument('--adapter', type=str, default='lora', help='dimension of latent factors')

    # Directory configuration conda activate torch37
    parser.add_argument('--datadir', type=str, required=False, default="./data", help="Data directory")
    parser.add_argument('--logdir', type=str, default='./log/', help='dataset used for training')
    parser.add_argument('--modeldir', type=str, default='./save/', help='dataset used for training')

    parser.add_argument('--local', type=bool, default=False, help='local or agg')

    # Computation configuration
    parser.add_argument('--device', type=str, default='2', help='The device to run the program')
    parser.add_argument('--init_seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_random_seed(args.init_seed)

    run(args)
