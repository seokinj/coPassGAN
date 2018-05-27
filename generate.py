import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import language_helpers
import tflib as lib
import tflib.plot
from models import *

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

BATCH_SIZE = 64
ITERS = 199000 # How many iterations to train for
SEQ_LEN = 12 # Sequence length in characters
DIM = 128 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 10 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 100000000#10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

DATA_DIR = '/Volumes/Transcend/text'

try:
	with open(DATA_DIR+'/3class12.dump', 'rb') as f:
		p = pickle.load(f)
		lines2 = p['lines']
		charmap2 = p['charmap']
		inv_charmap2 = p['inv_charmap']
except:
	print("Error")

#netG_A = torch.load('Generator_A.pt')
netG_B = torch.load('copy/Generator_B.pt')
#netD_A = torch.load('Discriminator_A.pt')
#netD_B = torch.load('Discriminator_B.pt')

def generate_samples(netG, charmap, inv_charmap, nv):
    samples1 = netG(noisev)
    samples1 = samples1.view(-1, SEQ_LEN, len(charmap))

    samples1 = samples1.cpu().data.numpy()
    samples1 = np.argmax(samples1, axis=2)
    
    decoded_samples1 = []
    for i in range(len(samples1)):
        decoded1 = []
        for j in range(len(samples1[i])):
            decoded1.append(inv_charmap[samples1[i][j]])
        decoded_samples1.append(tuple(decoded1))
    
    return decoded_samples1

noise = torch.randn(BATCH_SIZE, 128)

if use_cuda:
	noise = noise.cuda(gpu)

#noisev = autograd.Variable(noise, volatile=True)
with torch.no_grad():
    noisev = autograd.Variable(noise)

for i in range(1562500): # 10^8 
    #samples1.extend(generate_samples(netG_A, charmap1, inv_charmap1, noisev))
    samples2.extend(generate_samples(netG_B, charmap2, inv_charmap2, noisev))
      
with open(DATA_DIR+'/transfer_generate_B.txt', 'w') as f:
    for s in samples2:
        s = "".join(s)
        f.write(s + "\n")
