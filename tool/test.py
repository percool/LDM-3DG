    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='./logs/debug')
parser.add_argument('--condition', type=str, default='alpha')

parser.add_argument('--data_dir', type=str, default='../e3_diffusion_for_molecules/qm9/latent_diffusion/emb_2d_3d/')

parser.add_argument('--sample_number', type=int, default=100000)
parser.add_argument('--dim_condition', type=int, default=16)

# number of nodes for parallel training
parser.add_argument('--ddp_num_nodes', type=int, default=1)
# number of devices in each node for parallel training
parser.add_argument('--ddp_device', type=int, default=1)
args = parser.parse_args()
args.ddp_num_gpus = args.ddp_num_nodes * args.ddp_device

print(args)

import logging
logging.info('Beginning download of GDB9 dataset!')
print(1)
