#!/usr/bin/env python3
import os
import argparse
import glob

import torch
from visdom import Visdom

def main():
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

    visdom_addr = args.visdom_address
    visdom_port = args.visdom_port
    viz = Visdom(port=visdom_port, server=visdom_addr)

    paths = list(glob.glob(os.path.join(args.LOG_DIR, '*.zip')))
    paths.sort()

    for path in paths:
        log_name = os.path.basename(path)[-4]
        log = torch.jit.load(path, map_location=torch.device('cpu'))

        viz.histogram(
            X=log.means_gen.view([-1]),
        )


if __name__ == '__main__':
    main()
