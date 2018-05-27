import torch
import torch.autograd as autograd
import torch.nn as nn

BATCH_SIZE = 64
DIM = 128
SEQ_LEN = 12
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class Discriminator(nn.Module):

    def __init__(self, charmap1, charmap2):
        super(Discriminator, self).__init__()
        self.conv0_a = nn.Conv1d(len(charmap1), DIM, 1)
        self.conv0_b = nn.Conv1d(len(charmap2), DIM, 1)
        self.block1_a = ResBlock()
        self.block1_b = ResBlock()
        self.base2 = ResBlock()
        self.base3 = ResBlock()
        self.base4 = ResBlock()
        self.base5 = ResBlock()
        self.linear = nn.Linear(SEQ_LEN*DIM, 1)

    def forward(self, x_a, x_b):
        output_a = self.block1_a(self.conv0_a(x_a.transpose(1, 2))) 
        output_b = self.block1_b(self.conv0_b(x_b.transpose(1, 2)))
        #output = torch.cat((output_a, output_b), 0)
        output_a = self.base2(output_a)
        output_b = self.base2(output_b)
        output_a = self.base3(output_a)
        output_b = self.base3(output_b)
        output_a = self.base4(output_a)
        output_b = self.base4(output_b)
        output_a = self.base5(output_a)
        output_b = self.base5(output_b)
        output_a = output_a.view(-1, SEQ_LEN*DIM)
        output_b = output_b.view(-1, SEQ_LEN*DIM)
        output_a = self.linear(output_a)
        output_b = self.linear(output_b)
        return output_a, output_b

class Generator(nn.Module):

    def __init__(self, charmap1, charmap2):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(128, DIM*SEQ_LEN)
        self.base1 = ResBlock()
        self.base2 = ResBlock()
        self.base3 = ResBlock()
        self.base4 = ResBlock()
        self.block5_a = ResBlock()
        self.block5_b = ResBlock()        
        self.conv1_a = nn.Conv1d(DIM, len(charmap1), 1)
        self.conv1_b = nn.Conv1d(DIM, len(charmap2), 1)
        self.softmax_a = nn.Softmax()
        self.softmax_b = nn.Softmax()

    def forward(self, noise):
        output_a = self.fc1(noise)
        output_b = self.fc1(noise)
        output_a = output_a.view(-1, DIM, SEQ_LEN) # (BATCH_SIZE, DIM, SEQ_LEN)
        output_b = output_b.view(-1, DIM, SEQ_LEN)
        output_a = self.base1(output_a)
        output_b = self.base1(output_b)
        output_a = self.base2(output_a)
        output_b = self.base2(output_b)
        output_a = self.base3(output_a)
        output_b = self.base3(output_b)
        output_a = self.base4(output_a)
        output_b = self.base4(output_b)
        output_a = self.conv1_a(self.block5_a(output_a)).transpose(1,2)
        output_b = self.conv1_b(self.block5_b(output_b)).transpose(1,2)
        shape_a = output_a.size()
        shape_b = output_b.size()
        output_a = self.softmax_a(output_a.contiguous().view(BATCH_SIZE*SEQ_LEN, -1))
        output_b = self.softmax_b(output_b.contiguous().view(BATCH_SIZE*SEQ_LEN, -1))
        return output_a.view(shape_a), output_b.view(shape_b)
