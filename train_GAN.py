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

#fn1 = '1class8_train.txt'
fn2 = '3class12_train.txt'

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64 # Batch size
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


lib.print_model_settings(locals().copy())
"""
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
#print(charmap1)
"""
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

#table1 = np.arange(len(charmap1)).reshape(-1, 1) # (len(charmap), 1)
#one_hot1 = OneHotEncoder()
#one_hot1.fit(table1)

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

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(disc_interpolates.size()),create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

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

# ==================Definition End======================

#netG_A = Generator_A(charmap1)
netG_B = Generator_B(charmap2)
#netD_A = Discriminator_A(charmap1)
netD_B = Discriminator_B(charmap2)

if use_cuda:
    #netD_A = netD_A.cuda()
    netD_B = netD_B.cuda()
    #netG_A = netG_A.cuda()
    netG_B = netG_B.cuda()

#optimizerD_A = optim.Adam(netD_A.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD_B = optim.Adam(netD_B.parameters(), lr=1e-4, betas=(0.5, 0.9))
#optimizerG_A = optim.Adam(netG_A.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG_B = optim.Adam(netG_B.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

#data_A = inf_train_gen(lines1, charmap1)
data_B = inf_train_gen(lines2, charmap2)

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.

"""
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print ("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]
"""
"""
try:
    with open('data/char_ngram', 'rb') as f:
        p = pickle.load(f)
        validation_char_ngram_lms = p['vcnl']
        true_char_ngram_lms = p['tcnl']
except: 
    true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[10*batch_size:], tokenize=False) for i in range(4)]
    validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*batch_size], tokenize=False) for i in range(4)]
    for i in range(4):
        print("validation set JSD for n=%d: %d" % (i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
    true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]
    with open('data/char_ngram', 'wb') as f:
        pickle.dump({'tcnl':true_char_ngram_lms, 'vcnl':validation_char_ngram_lms}, f)
"""

print("Start")
for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    
    #for p in netD_A.parameters():  # reset requires_grad
        #p.requires_grad = True  # they are set to False below in netG update

    for p in netD_B.parameters():  # reset requires_grad
        p.requires_grad = True

    for iter_d in range(CRITIC_ITERS):
        #_data_A = next(data_A)
        _data_B = next(data_B)
        
        #data_one_hot_A = one_hot1.transform(_data_A.reshape(-1, 1)).toarray().reshape(BATCH_SIZE, -1, len(charmap1))
        data_one_hot_B = one_hot2.transform(_data_B.reshape(-1, 1)).toarray().reshape(BATCH_SIZE, -1, len(charmap2))
        #print data_one_hot.shape
        #real_data_A = torch.Tensor(data_one_hot_A)
        real_data_B = torch.Tensor(data_one_hot_B)
        if use_cuda:
            #real_data_A = real_data_A.cuda()
            real_data_B = real_data_B.cuda()
        #real_data_A_v = autograd.Variable(real_data_A)
        real_data_B_v = autograd.Variable(real_data_B)

        #netD_A.zero_grad()
        netD_B.zero_grad()

        # train with real
        #D_real_A = netD_A(real_data_A_v)
        #D_real_A = D_real_A.mean()
        D_real_B = netD_B(real_data_B_v)
        D_real_B = D_real_B.mean()
        # print D_real
        # TODO: Waiting for the bug fix from pytorch
        #D_real_A.backward(mone)
        D_real_B.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda()
        #noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        with torch.no_grad():
            noisev = autograd.Variable(noise)        
        #fake_A = autograd.Variable(netG_A(noisev).data)
        fake_B = autograd.Variable(netG_B(noisev).data)

        #inputAv = fake_A
        #D_fake_A = netD_A(inputAv)
        #D_fake_A = D_fake_A.mean()
        inputBv = fake_B
        D_fake_B = netD_B(inputBv)
        D_fake_B = D_fake_B.mean()
        # TODO: Waiting for the bug fix from pytorch
        #D_fake_A.backward(one)
        D_fake_B.backward(one)

        # train with gradient penalty
        #gradient_penalty_A = calc_gradient_penalty(netD_A, real_data_A_v.data, fake_A.data)
        #gradient_penalty_A.backward(retain_graph=True)
        gradient_penalty_B = calc_gradient_penalty(netD_B, real_data_B_v.data, fake_B.data)
        gradient_penalty_B.backward(retain_graph=True)

        #D_cost_A = D_fake_A - D_real_A + gradient_penalty_A
        #Wasserstein_D_A = D_real_A - D_fake_A
        D_cost_B = D_fake_B - D_real_B + gradient_penalty_B
        Wasserstein_D_B = D_real_B - D_fake_B

        #optimizerD_A.step()
        optimizerD_B.step()

    ############################
    # (2) Update G network
    ###########################
    #for p in netD_A.parameters():
        #p.requires_grad = False  # to avoid computation
    for p in netD_B.parameters():
        p.requires_grad = False

    #netG_A.zero_grad()
    netG_B.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)

    #fake_A = netG_A(noisev)
    #G_A = netD_A(fake_A)
    #G_A = G_A.mean()
    #G_A.backward(mone)
    #G_cost_A = -G_A

    fake_B = netG_B(noisev)
    G_B = netD_B(fake_B)
    G_B = G_B.mean()
    G_B.backward(mone)
    G_cost_B = -G_B

    #optimizerG_A.step()
    optimizerG_B.step()

    # Write logs and save samples
    lib.plot.plot('tmp/lang2/time', time.time() - start_time)
    #lib.plot.plot('tmp/lang/train Discriminator_A cost', D_cost_A.cpu().data.numpy())
    lib.plot.plot('tmp/lang2/train Discriminator_B cost', D_cost_B.cpu().data.numpy())
    #lib.plot.plot('tmp/lang/train Generator_A cost', G_cost_A.cpu().data.numpy())
    lib.plot.plot('tmp/lang2/train Generator_B cost', G_cost_B.cpu().data.numpy())
    #lib.plot.plot('tmp/lang/wasserstein distance A', Wasserstein_D_A.cpu().data.numpy())
    lib.plot.plot('tmp/lang2/wasserstein distance B', Wasserstein_D_B.cpu().data.numpy())

                     
    if iteration % 100 == 99:
        print(iteration+1)
        #samples1 = []
        samples2 = []
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        #noisev = autograd.Variable(noise, volatile=True)
        with torch.no_grad():
            noisev = autograd.Variable(noise)
        for i in range(10):
            #samples1.extend(generate_samples(netG_A, charmap1, inv_charmap1, noisev))
            samples2.extend(generate_samples(netG_B, charmap2, inv_charmap2, noisev))
        """
        with open(DATA_DIR+'/test/generate_A_{}.txt'.format(iteration+1), 'w') as f:    
            for s in samples1:
                s = "".join(s)
                f.write(s+"\n")
        """ 
        with open(DATA_DIR+'/test/generate_B_{}.txt'.format(iteration+1), 'w') as f:
            for s in samples2:
                s = "".join(s)
                f.write(s + "\n")
        
    if iteration % 1000 == 999:
        #torch.save(netG_A.state_dict(),'Generator_A.pt')
        #torch.save(netD_A.state_dict(),'Discriminator_A.pt')
        torch.save(netG_B.state_dict(), 'Generator_B.pt')
        torch.save(netD_B.state_dict(), 'Discriminator_B.pt')

    if iteration % 100 == 99:
        lib.plot.flush()

    lib.plot.tick()
