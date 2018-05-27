import os, sys
sys.path.append(os.getcwd())
import time
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
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = './data'
RESULT_DIR = './result'
CP_DIR = './checkpoint'

fn1 = '1class8_train.txt'
fn2 = '3class12_train.txt'

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64 
ITERS = 400000
SEQ_LEN = 12
DIM = 128 
CRITIC_ITERS = 10
LAMBDA = 10 
MAX_N_EXAMPLES = 100000000

lib.print_model_settings(locals().copy())
try:
    with open(DATA_DIR+'/1class8.dump', 'rb') as f:
        p = pickle.load(f)
        lines1 = p['lines']
        charmap1 = p['charmap']
        inv_charmap1 = p['inv_charmap']
except:
    lines1, charmap1, inv_charmap1 = language_helpers.load_dataset(
        max_length=SEQ_LEN,
        max_n_examples=MAX_N_EXAMPLES,
        data_dir=DATA_DIR,
        fn = fn1
    )
    with open(DATA_DIR+"/1class8.dump", 'wb') as f:
        pickle.dump({'lines':lines1, 'charmap':charmap1, 'inv_charmap':inv_charmap1}, f)

try:                  
    with open(DATA_DIR+'/3class12.dump', 'rb') as f:
        p = pickle.load(f)
        lines2 = p['lines']
        charmap2 = p['charmap']
        inv_charmap2 = p['inv_charmap']
except:
    lines2, charmap2, inv_charmap2 = language_helpers.load_dataset(
        max_length=SEQ_LEN,
        max_n_examples=MAX_N_EXAMPLES,
        data_dir=DATA_DIR,
        fn = fn2
    )
    with open(DATA_DIR+"/3class12.dump", 'wb') as f:
        pickle.dump({'lines':lines2, 'charmap':charmap2, 'inv_charmap':inv_charmap2}, f)

table1 = np.arange(len(charmap1)).reshape(-1, 1) # (len(charmap), 1)
one_hot1 = OneHotEncoder()
one_hot1.fit(table1)

table2 = np.arange(len(charmap2)).reshape(-1, 1) # (len(charmap), 1)
one_hot2 = OneHotEncoder()
one_hot2.fit(table2)

# ==================Definition Start======================
# Dataset iterator
def inf_train_gen(lines, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )

def calc_gradient_penalty(net, realA, realB, fakeA, fakeB): # [128, 12, 95], [128, 12, 95]
	alpha = torch.randn(BATCH_SIZE, 1, 1)
	alpha2 = torch.randn(BATCH_SIZE, 1, 1)
	interpolates1 = alpha * fakeA + ((1-alpha)*realA) # [128, 12, 95]
	interpolates2 = alpha * fakeB + ((1-alpha2)*realB)
	if use_cuda:
		interpolates1 = interpolates1.cuda()
		interpolates2 = interpolates2.cuda()
	interpolates1 = autograd.Variable(interpolates1, requires_grad=True)
	interpolates2 = autograd.Variable(interpolates2, requires_grad=True) 

	disc_interpolates1, disc_interpolates2 = net(interpolates1, interpolates2)
	
	gradients1 = autograd.grad(outputs=disc_interpolates1, inputs=interpolates1, grad_outputs=torch.ones(disc_interpolates1.size()).cuda() if use_cuda else torch.ones(disc_interpolates1.size()),create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients2 = autograd.grad(outputs=disc_interpolates2, inputs=interpolates2, grad_outputs=torch.ones(disc_interpolates2.size()).cuda() if use_cuda else torch.ones(disc_interpolates2.size()),create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty1 = (((gradients1.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA).view(1)
	gradient_penalty2 = (((gradients2.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA).view(1)
 	
	return gradient_penalty1, gradient_penalty2

def generate_samples(net, nv):
    samp1, samp2 = net(nv)
    samp1 = samp1.view(-1, SEQ_LEN, len(charmap1))
    samp2 = samp2.view(-1, SEQ_LEN, len(charmap2))

    samp1 = samp1.cpu().data.numpy()
    samp1 = np.argmax(samp1, axis=2)
    
    samp2 = samp2.cpu().data.numpy()
    samp2 = np.argmax(samp2, axis=2)

    decoded_samples1 = []
    decoded_samples2 = []

    for i in range(len(samp1)):
        decoded1 = []
        for j in range(len(samp1[i])):
            decoded1.append(inv_charmap1[samp1[i][j]])
        decoded_samples1.append(tuple(decoded1))

    for i in range(len(samp2)):
        decoded2 = []
        for j in range(len(samp2[i])):
            decoded2.append(inv_charmap2[samp2[i][j]])
        decoded_samples2.append(tuple(decoded2))
    
    return decoded_samples1, decoded_samples2

# ==================Definition End======================


netG = Generator(charmap1, charmap2)
netD = Discriminator(charmap1, charmap2)

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=2*1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=2*1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data_A = inf_train_gen(lines1, charmap1)
data_B = inf_train_gen(lines2, charmap2)


print("Start")
for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        _data_A = next(data_A)
        _data_B = next(data_B)
        
        data_one_hot_A = one_hot1.transform(_data_A.reshape(-1, 1)).toarray().reshape(BATCH_SIZE, -1, len(charmap1))
        data_one_hot_B = one_hot2.transform(_data_B.reshape(-1, 1)).toarray().reshape(BATCH_SIZE, -1, len(charmap2))
        #print data_one_hot.shape
        real_data_A = torch.Tensor(data_one_hot_A)
        real_data_B = torch.Tensor(data_one_hot_B)
        if use_cuda:
            real_data_A = real_data_A.cuda()
            real_data_B = real_data_B.cuda()
        real_data_A_v = autograd.Variable(real_data_A)
        real_data_B_v = autograd.Variable(real_data_B)

        netD.zero_grad()
        
        # train with real
        D_real_A, D_real_B = netD(real_data_A_v, real_data_B_v)
        D_real_A = D_real_A.mean()
        D_real_B = D_real_B.mean()
        
        # print D_real
        # TODO: Waiting for the bug fix from pytorch
        D_real_A.backward(mone)
        D_real_B.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda()
        with torch.no_grad():
            noisev = autograd.Variable(noise)  # totally freeze netG
        fake_a, fake_b = netG(noisev)
        fake_a = autograd.Variable(fake_a.data)
        fake_b = autograd.Variable(fake_b.data)	

        inputAv = fake_a
        inputBv = fake_b
        D_fake_A, D_fake_B = netD(inputAv, inputBv)
        D_fake_A = D_fake_A.mean()
        D_fake_B = D_fake_B.mean()

        # TODO: Waiting for the bug fix from pytorch
        D_fake_A.backward(one)
        D_fake_B.backward(one)
        
        # train with gradient penalty
        #real_data = torch.cat((real_data_A_v.data, real_data_B_v.data), 0)
        #fake_data = torch.cat((fake_a.data, fake_b.data), 0)
        gradient_penalty_A, gradient_penalty_B = calc_gradient_penalty(netD, real_data_A_v.data, real_data_B_v.data, fake_a.data, fake_b.data)
        gradient_penalty_A.backward(retain_graph=True)
        gradient_penalty_B.backward(retain_graph=True)

        D_cost_A = D_fake_A - D_real_A + gradient_penalty_A
        D_cost_B = D_fake_B - D_real_B + gradient_penalty_B
        Wasserstein_D_A = D_real_A - D_fake_A
        Wasserstein_D_B = D_real_B - D_fake_B

        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
   
    netG.zero_grad()
    
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake_a, fake_b = netG(noisev)
    inputAv = fake_a
    inputBv = fake_b   

    G_A, G_B = netD(inputAv, inputBv)
    G_A = G_A.mean()
    G_B = G_B.mean()
    G_A.backward(mone)
    G_B.backward(mone)
    G_cost_A = -G_A
    G_cost_B = -G_B

    optimizerG.step()
    
    # Write logs and save samples
    lib.plot.plot('tmp/lang/time', time.time() - start_time)
    lib.plot.plot('tmp/lang/train Discriminator A cost', D_cost_A.cpu().data.numpy())
    lib.plot.plot('tmp/lang/train Generator A cost', G_cost_A.cpu().data.numpy())
    lib.plot.plot('tmp/lang/wasserstein A distance', Wasserstein_D_A.cpu().data.numpy())

    lib.plot.plot('tmp/lang/train Discriminator B cost', D_cost_B.cpu().data.numpy())
    lib.plot.plot('tmp/lang/train Generator B cost', G_cost_B.cpu().data.numpy())
    lib.plot.plot('tmp/lang/wasserstein B distance', Wasserstein_D_B.cpu().data.numpy())

                     
    if iteration % 100 == 99:
        print(iteration+1)
        
    if iteration % 100 == 99:
        torch.save(netG.state_dict(), CP_DIR+'/Generator_{}.pt'.format(iteration+1))
        torch.save(netD.state_dict(), CP_DIR+'/Discriminator_{}.pt'.format(iteration+1))

        l = 5
        for i in range(l):
            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
            with torch.no_grad():
                noisev = autograd.Variable(noise)
            samples1, samples2 = generate_samples(netG, noisev)
            with open(RESULT_DIR+'/cogan/cogan_A_{}.txt'.format(iteration+1), 'a') as f:
                for s in samples1:
                    s = "".join(s)
                    f.write(s+'\n')
            with open(RESULT_DIR+'/cogan/cogan_B_{}.txt'.format(iteration+1), 'a') as f:
                for s in samples2:
                    s = "".join(s)
                    f.write(s+'\n')
    if iteration % 100 == 99:
        lib.plot.flush()

    lib.plot.tick()
