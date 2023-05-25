import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from mixture_of_mvns import MixtureOfMVNs
from models_amortized_clustering import EmbedderMoG
from mvn_diag import MultivariateNormalDiag
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train', help='{"bench","train","test","plot"}')
parser.add_argument('--num_bench', type=int, default=100, help='#batches for benchmark test data, each batch has "B" sets of equal cardinality')
parser.add_argument('--B', type=int, default=10, help='batch size; # of MoG samples/datasets of equal cardinality')
parser.add_argument('--N_min', type=int, default=100, help='minimum set cardinality')
parser.add_argument('--N_max', type=int, default=500, help='maximum set cardinality')
parser.add_argument('--K', type=int, default=4, help='mixture order')
parser.add_argument('--gpu', type=str, default='0', help='gpu id to use')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--run_name', type=str, default='trial', help='remark')
parser.add_argument('--num_steps', type=int, default=50000, help='total number of batch iters')
parser.add_argument('--test_freq', type=int, default=200, help='do testing every this batch iters')
parser.add_argument('--save_freq', type=int, default=400, help='save model every this batch iters')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--num_proto', type=int, default=4, help='number of prototypes (p)')
parser.add_argument('--num_ems', type=int, default=3, help='number of EM steps (k)')
parser.add_argument('--num_heads', type=int, default=5, help='number of priors thetas (H)')
parser.add_argument('--tau', type=float, default=10.0, help='prior impact in diem')
parser.add_argument('--dim_feat', type=int, default=128, help='feature dimension')
parser.add_argument('--out_type', type=str, default='select_best2', help='output type')
args = parser.parse_args()

B = args.B
N_min = args.N_min
N_max = args.N_max
K = args.K
D = 2  # dimensionality of individual set elements
mvn = MultivariateNormalDiag(D)
mog = MixtureOfMVNs(mvn)
dim_output = 2 * D

distr_emb_args = {
    "dh": 128, "dout": 64, "num_eps": 5,
    "layers1": [128], "nonlin1": 'relu', "layers2": [128], "nonlin2": 'relu',
}

net = EmbedderMoG(D, K, out_type=args.out_type,
    num_proto=args.num_proto, num_ems=args.num_ems,
    dim_feat=args.dim_feat, num_heads=args.num_heads, tau=args.tau,
    distr_emb_args=distr_emb_args).cuda()

benchfile = os.path.join('benchmark', 'mog_{:d}_Nmin%s_Nmax%s.pkl'.format(K) % (args.N_min, args.N_max))
save_dir = os.path.join('results', 'synth_clustering',args.run_name)


def generate_benchmark():
    if not os.path.isdir('benchmark'):
        os.makedirs('benchmark')
    N_list = np.random.randint(N_min, N_max, args.num_bench)
    data = []
    ll = 0.
    for N in tqdm(N_list):
        X, labels, pi, params = mog.sample(B, N, K, return_gt=True)
        ll += mog.log_prob(X, pi, params).item()
        data.append(X)
    bench = [data, ll / args.num_bench]
    torch.save(bench, benchfile)


def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isfile(benchfile):
        generate_benchmark()

    bench = torch.load(benchfile)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(
        logging.FileHandler(
            os.path.join(save_dir, 'train_' + time.strftime('%Y%m%d-%H%M') + '.log'),
            mode='w'
        )
    )
    logger.info(str(args) + '\n')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    tick = time.time()
    for t in range(1, args.num_steps + 1):
        if t == int(0.5 * args.num_steps):
            optimizer.param_groups[0]['lr'] *= 0.1

        net.train()

        optimizer.zero_grad()
        N = np.random.randint(N_min, N_max)
        X = mog.sample(B, N, K)
        ll = mog.log_prob(X, *net(X))
        loss = -ll
        loss.backward()
        optimizer.step()

        if t % args.test_freq == 0:
            line = 'step {}, lr {:.3e}, '.format(t, optimizer.param_groups[0]['lr'])
            line += test(bench, verbose=False)
            line += ' ({:.3f} secs)'.format(time.time() - tick)
            tick = time.time()
            logger.info(line)

        if t % args.save_freq == 0:
            torch.save({'state_dict': net.state_dict()}, os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict': net.state_dict()}, os.path.join(save_dir, 'model.tar'))


def test(bench, verbose=True):
    net.eval()

    data, oracle_ll = bench

    avg_ll = 0.
    for X in data:
        X = X.cuda()
        avg_ll += mog.log_prob(X, *net(X)).item()
    avg_ll /= len(data)

    line = 'test ll {:.4f} (oracle {:.4f})'.format(avg_ll, oracle_ll)

    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(args.run_name)
        logger.addHandler(logging.FileHandler(os.path.join(save_dir, 'test.log'), mode='w'))
        logger.info(line)

    return line


if __name__ == '__main__':
    if args.mode == 'bench':
        generate_benchmark()
    elif args.mode == 'train':
        train()
    elif args.mode == 'test':
        bench = torch.load(benchfile)
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        net.load_state_dict(ckpt['state_dict'])
        test(bench)
