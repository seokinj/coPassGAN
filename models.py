import torch
import torch.autograd as autograd
import torch.nn as nn

BATCH_SIZE = 64
DIM = 128
SEQ_LEN = 12

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

G_block_share1 = ResBlock()
G_block_share2 = ResBlock()

D_block_share4 = ResBlock()
D_block_share5 = ResBlock()

class Generator_A(nn.Module):

    def __init__(self, charmap):
        super(Generator_A, self).__init__()

        self.fc1 = nn.Linear(128, DIM*SEQ_LEN)
        self.blockA3 = ResBlock()
        self.blockA4 = ResBlock()
        self.blockA5 = ResBlock()
        
        self.conv1 = nn.Conv1d(DIM, len(charmap), 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, DIM, SEQ_LEN) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = G_block_share1(output)
        output = G_block_share2(output)
        output = self.blockA3(output)
        output = self.blockA4(output)
        output = self.blockA5(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(BATCH_SIZE*SEQ_LEN, -1)
        output = self.softmax(output)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Generator_B(nn.Module):

    def __init__(self, charmap):
        super(Generator_B, self).__init__()

        self.fc1 = nn.Linear(128, DIM*SEQ_LEN)
        self.blockB3 = ResBlock()
        self.blockB4 = ResBlock()
        self.blockB5 = ResBlock()
        
        self.conv1 = nn.Conv1d(DIM, len(charmap), 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, DIM, SEQ_LEN) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = G_block_share1(output)
        output = G_block_share2(output)
        output = self.blockB3(output)
        output = self.blockB4(output)
        output = self.blockB5(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(BATCH_SIZE*SEQ_LEN, -1)
        output = self.softmax(output)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Discriminator_A(nn.Module):

    def __init__(self, charmap):
        super(Discriminator_A, self).__init__()

        self.blockA1 = ResBlock()
        self.blockA2 = ResBlock()
        self.blockA3 = ResBlock()

        self.conv1d = nn.Conv1d(len(charmap), DIM, 1)
        self.linear = nn.Linear(SEQ_LEN*DIM, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.blockA1(output)
        output = self.blockA2(output)
        output = self.blockA3(output)
        output = D_block_share4(output)
        output = D_block_share5(output)        
        output = output.view(-1, SEQ_LEN*DIM)
        output = self.linear(output)
        return output

class Discriminator_B(nn.Module):

    def __init__(self, charmap):
        super(Discriminator_B, self).__init__()

        self.blockB1 = ResBlock()
        self.blockB2 = ResBlock()
        self.blockB3 = ResBlock()

        self.conv1d = nn.Conv1d(len(charmap), DIM, 1)
        self.linear = nn.Linear(SEQ_LEN*DIM, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.blockB1(output)
        output = self.blockB2(output)
        output = self.blockB3(output)
        output = D_block_share4(output)
        output = D_block_share5(output)        
        output = output.view(-1, SEQ_LEN*DIM)
        output = self.linear(output)
        return output
