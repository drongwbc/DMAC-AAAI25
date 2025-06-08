import argparse
from train import train

if __name__ == "__main__":
    # Demo for PIE
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--name", type=str, default='PIE')
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--hidden_size", type=int,
                        default=256)
    parser.add_argument("--emblem_size", type=int,
                        default=64)
    parser.add_argument("--max_preEpoch", type=int, default=1000) # optional (allow 0 as input)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1e+3)
    parser.add_argument("--beta", type=float, default=1e+3)
    parser.add_argument("--lr1", type=float, default=1e-3) # Learning rate for pretrain
    parser.add_argument("--lr2", type=float, default=5e-3) # Learning rate for model train
    args = parser.parse_args()

    train(args)


