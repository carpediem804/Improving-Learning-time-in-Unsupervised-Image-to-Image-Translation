import argparse
import os
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from network import Generator

parser = argparse.ArgumentParser(description='DiscoGAN in One Code')

# Task
parser.add_argument('--task', required=True, help='task or root name')

# Hyper-parameters
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')

# misc
parser.add_argument('--model_path', type=str, default='./models-residual-시간2')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./result나와랏')  # Results

##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        self.task = opt.task
        self.transformP = transforms.Compose([transforms.Scale((128, 64)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.transformS = transforms.Compose([transforms.Scale((64, 64)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.image_len = None

        self.dir_base = './datasets'
        if self.task.startswith('edges2'):
            self.root = os.path.join(self.dir_base, self.task)
            self.dir_AB = os.path.join(self.root, 'val')  # ./maps/train
            self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))
            self.image_len = len(self.image_paths)

        elif self.task == 'handbags2shoes': # handbags2shoes
            self.rootA = os.path.join(self.dir_base, 'edges2handbags')
            self.rootB = os.path.join(self.dir_base, 'edges2shoes')
            self.dir_A = os.path.join(self.rootA, 'val')
            self.dir_B = os.path.join(self.rootB, 'val')
            self.image_paths_A = list(map(lambda x: os.path.join(self.dir_A, x), os.listdir(self.dir_A)))
            self.image_paths_B = list(map(lambda x: os.path.join(self.dir_B, x), os.listdir(self.dir_B)))
            self.image_len = min(len(self.image_paths_A), len(self.image_paths_B))

        else: # facescrubs
            self.root = os.path.join(self.dir_base, 'facescrub')
            #self.rootA = os.path.join(self.root, 'actors')
            #self.rootB = os.path.join(self.root, 'actresses')
            self.dir_A = os.path.join(self.root, 'male') # You Should make your OWN Validation Set
            self.dir_B = os.path.join(self.root, 'female')
            #self.dir_A = os.path.join(self.root, 'trainA')
            #self.dir_B = os.path.join(self.root, 'trainB')
            self.image_paths_A = list(map(lambda x: os.path.join(self.dir_A, x), os.listdir(self.dir_A)))
            self.image_paths_B = list(map(lambda x: os.path.join(self.dir_B, x), os.listdir(self.dir_B)))
            self.image_len = min(len(self.image_paths_A), len(self.image_paths_B))

    def __getitem__(self, index):
        if self.task.startswith('edges2'):
            AB_path = self.image_paths[index]
            AB = Image.open(AB_path).convert('RGB')
            AB = self.transformP(AB)

            w_total = AB.size(2)
            w = int(w_total / 2)

            A = AB[:, :64, :64]
            B = AB[:, :64, w:w + 64]

        elif self.task == 'handbags2shoes': # handbags2shoes
            A_path = self.image_paths_A[index]
            B_path = self.image_paths_B[index]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')

            A = self.transformP(A)
            B = self.transformP(B)

            w_total = A.size(2)
            w = int(w_total / 2)

            A = A[:, :64, w:w+64]
            B = B[:, :64, w:w+64]

        else: # Facescrubs
            A_path = self.image_paths_A[index]
            B_path = self.image_paths_B[index]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')

            A = self.transformS(A)
            B = self.transformS(B)

        return {'A': A, 'B': B}

    def __len__(self):
        return self.image_len

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

######################### Main Function
def main():
    # Pre-settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=2)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    g_pathAtoB = os.path.join(args.model_path, 'generatorAtoB-500.pkl' )
    g_pathBtoA = os.path.join(args.model_path, 'generatorBtoA-500.pkl' )

    generator_AtoB = Generator()
    generator_BtoA = Generator()

    generator_AtoB.load_state_dict(torch.load(g_pathAtoB))
    generator_AtoB.eval()

    generator_BtoA.load_state_dict(torch.load(g_pathBtoA))
    generator_BtoA.eval()

    if torch.cuda.is_available():
        generator_AtoB = generator_AtoB.cuda()
        generator_BtoA = generator_BtoA.cuda()

    """Train generator and discriminator."""
    total_step = len(data_loader) # For Print Log
    iter = 0
    for i, sample in enumerate(data_loader):
        input_A = sample['A']
        input_B = sample['B']

        A = to_variable(input_A)
        B = to_variable(input_B)

        # ===================== Forward =====================#
        A_to_B = generator_AtoB(A)
        B_to_A = generator_BtoA(B)

        A_to_B_to_A = generator_BtoA(A_to_B)
        B_to_A_to_B = generator_AtoB(B_to_A)

        # print the log info
        print('Validation [%d/%d]' % (i + 1, total_step))

        # save the sampled images
        res1 = torch.cat((torch.cat((A, A_to_B), dim=3), A_to_B_to_A), dim=3)
        res2 = torch.cat((torch.cat((B, B_to_A), dim=3), B_to_A_to_B), dim=3)
        res = torch.cat((res1, res2), dim=2)
        torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, 'Generated-%d.png' % (i + 1)))


if __name__ == "__main__":
    main()
