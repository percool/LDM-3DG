import argparse
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pdb
import os, sys
from hgraph.hgnn import HierVAE
from hgraph.vocab import PairVocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./')

    parser.add_argument('--sample_number', type=int, default=10000)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)
    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # # load latent samples -- Is this originally include latent 3d and 2d?
    # samples = torch.load('./samples_latent/sample_z.pt') # [:10000]

    # TODO: directly load ground truth smile to the decoder
    
    samples = torch.load('./samples_latent_dock/emb2ds.pt')
    n1, n3 = samples.shape
    n2=1000 # num_sample in sample_z.py

    samples=samples.unsqueeze(1)
    # pdb.set_trace()
    # samples=np.expand_dims(samples, 1) # in sample_z, emb2ds.pt does not copy n2 times in the 2nd dimension.
    # samples=np.repeat(samples,n2,axis=1)

    samples=samples.expand(-1,n2,-1)
    # pdb.set_trace()
    samples = samples.reshape((n1*n2, n3))[:, :250]

    # 2. decode G~p(G|z)
    args3 = torch.load('../AE_topo_weights_and_data/args_new.pt')
    # vocab = [x.strip('\r\n ').split() for x in open('../AE_topo_weights_and_data/vocab_pocket_aware.txt')]
    vocab = [x.strip('\r\n ').split() for x in open('../AE_Topology/vocab_pocket_aware.txt')]
    args3.vocab = PairVocab(vocab)
    model = HierVAE(args3).to(device)
    ckpt = torch.load('../AE_topo_weights_and_data/pocket_pretrained/last.ckpt')
    state_dict = {k[6:]:v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    dataset = torch.utils.data.TensorDataset(samples)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64*8*4, shuffle=False, num_workers=8)
    smiles = []
    for batch in tqdm(dataloader):
        batch = batch[0].to(device)
        with torch.no_grad():
            ss = model.decode(batch)
            smiles += ss

    torch.save(smiles, './samples_latent_dock/sample_smiles.pt')

