import numpy as np
import torch
from torch.utils.data import DataLoader
# import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import cvxpy as cp
import numpy as np
import random

if __name__ == "__main__":
        """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),arg_mpLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    np.random.seed(args.data_seed)
    nsim = 10000  # number of datapoints: increase sample density for more robust results
    samples = {"a1": np.random.uniform(low=0.1, high=1.5, size=(nsim, 1)),
               "a2": np.random.uniform(low=0.1, high=2.0, size=(nsim, 1)),
               "p1": np.random.uniform(low=5.0, high=10.0, size=(nsim, 1)),
               "p2": np.random.uniform(low=5.0, high=10.0, size=(nsim, 1)),
               "p3": np.random.uniform(low=5.0, high=10.0, size=(nsim, 1))}
    nstep_data, dims = get_dataloaders(samples)
    train_data, dev_data, test_data = nstep_data
