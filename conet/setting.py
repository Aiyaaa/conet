'''
Configs for training & testing
Written by Whalechen
'''

import argparse
from os import path

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("network_trainer")
   
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument(
        '--model',
        default=None,
        type=str,
        help='Net work')
    parser.add_argument(
        '--num_workers',
        default=6,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=10, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )
    parser.add_argument(
        '--tag',
        default='',
        type=str,
        help='save_tag'
    )
    parser.add_argument(
        '--nseg',
        default=0,
        type=int,
        help='seg of sp')
    parser.add_argument(
        '--kl',
        default=1.,
        type=float,
        help='float weight')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_run", help="continue training",
                        action="store_true")
    parser.set_defaults(no_cuda=False)
    
    args = parser.parse_args()
    if len(args.tag):
        # suf = f'_
        pass
    # args.save_folder = "./trails/models" + args.tag

    args.save_folder = path.join("./trails",  args.network_trainer) + args.tag
    
    return args
