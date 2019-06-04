#!/usr/bin/env python3
import os
import argparse
import glob
import time

import torch
from visdom import Visdom

def main():
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('LOG_DIR')
    arg_parser.add_argument(
        '--visdom-address',
        default='localhost'
    )
    arg_parser.add_argument(
        '--visdom-port',
        type=int,
        default=8097
    )
    args = arg_parser.parse_args()

    # Connect to Visdom server
    visdom_addr = args.visdom_address
    visdom_port = args.visdom_port
    viz = Visdom(port=visdom_port, server=visdom_addr)

    # Load log data
    paths = list(glob.glob(os.path.join(args.LOG_DIR, '*.zip')))
    paths.sort()
    elbo_loss_x = list()
    elbo_loss_y = list()
    target_mse_x = list()
    target_mse_y = list()

    for path in paths:
        log_name = os.path.basename(path)[:-4]
        step, ts = log_name.split('-')
        step = int(step)
        ts = int(ts)
        log = torch.jit.load(path, map_location=torch.device('cpu'))

        elbo_loss_x.append(step)
        elbo_loss_y.append(log.elbo_loss)

        target_mse_x.append(step)
        target_mse_y.append(log.target_mse)

        # Uncomment this to plot histogram
        # viz.histogram(X=log.stds_gen.view([-1]))
        # viz.images(log.means_target, opts=dict(title=log_name))

    # Plot ELBO loss and target MSE curve
    elbo_loss_y = torch.stack(elbo_loss_y).detach().numpy()
    viz.line(X=elbo_loss_x, Y=elbo_loss_y)

    target_mse_y = torch.stack(target_mse_y).detach().numpy()
    viz.line(X=target_mse_x, Y=target_mse_y)


if __name__ == '__main__':
    main()
