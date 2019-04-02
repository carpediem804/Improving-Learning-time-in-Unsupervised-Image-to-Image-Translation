import argparse
import os
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from network import *
from itertools import chain
import time
torch.cuda.set_device(2)

parser = argparse.ArgumentParser(description='DiscoGAN in One Code')

# Task
parser.add_argument('--task', required=True, help='task or root name')

# Hyper-parameters
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

# DiscoGAN Parameters
parser.add_argument('--decay_gan_loss', type=int, default=10000)
parser.add_argument('--starting_rate', type=float, default=0.01)
parser.add_argument('--changed_rate', type=float, default=0.5)

# misc
parser.add_argument('--model_path', type=str, default='./models-residual-시간2')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./results-residual-시간2')  # Results
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=25)
parser.add_argument('--num_workers', type=int, default=2)

##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        self.task = opt.task
        self.transformP = transforms.Compose([transforms.Scale((128, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.transformS = transforms.Compose([transforms.Scale((128, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.image_len = None

        self.dir_base = './datasets'
        if self.task.startswith('edges2'):
            self.root = os.path.join(self.dir_base, self.task)
            self.dir_AB = os.path.join(self.root, 'train')  # ./maps/train
            self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))
            self.image_len = len(self.image_paths)

        elif self.task == 'handbags2shoes': # handbags2shoes
            self.rootA = os.path.join(self.dir_base, 'edges2handbags')
            self.rootB = os.path.join(self.dir_base, 'edges2shoes')
            self.dir_A = os.path.join(self.rootA, 'train')
            self.dir_B = os.path.join(self.rootB, 'train')
            self.image_paths_A = list(map(lambda x: os.path.join(self.dir_A, x), os.listdir(self.dir_A)))
            self.image_paths_B = list(map(lambda x: os.path.join(self.dir_B, x), os.listdir(self.dir_B)))
            self.image_len = min(len(self.image_paths_A), len(self.image_paths_B))

        else: # facescrubs
            self.root = os.path.join(self.dir_base, 'facescrub')
            self.rootA = os.path.join(self.root, 'actors')
            self.rootB = os.path.join(self.root, 'actresses')
            self.dir_A = os.path.join(self.rootA, 'face')
            self.dir_B = os.path.join(self.rootB, 'face')
            #self.dir_A = os.path.join(self.root, 'Cat')
            #self.dir_B = os.path.join(self.root, 'Dog')
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
        #x = x.cuda()
        torch.cuda.set_device(2)
        x = x.cuda()
    return Variable(x)


##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

##### Helper Functions for GAN Loss (4D Loss Comparison)
def GAN_Loss(input, target, criterion):
    if target == True:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(1.0)
        labels = Variable(tmp_tensor, requires_grad=False)
    else:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(0.0)
        labels = Variable(tmp_tensor, requires_grad=False)

    if torch.cuda.is_available():
        torch.cuda.set_device(2)
        labels = labels.cuda()

    return criterion(input, labels)

##### Feature Loss from Author's Code
def Feature_Loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion(l2, Variable(torch.ones(l2.size())).cuda())
        losses += loss

    return losses

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
                                  num_workers=args.num_workers)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    generator_AtoB = Generator()
    generator_BtoA = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    # Losses
    criterionGAN = nn.BCELoss()
    criterionRecon = nn.MSELoss()
    criterionFeature = nn.HingeEmbeddingLoss()

    # Optimizers
    g_params = chain(generator_AtoB.parameters(), generator_BtoA.parameters())
    d_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    g_optimizer = optim.Adam(g_params, args.lr, [args.beta1, args.beta2], weight_decay = 0.00001)
    d_optimizer = optim.Adam(d_params, args.lr, [args.beta1, args.beta2], weight_decay = 0.00001)

    if torch.cuda.is_available():
        torch.cuda.set_device(2)
        print(torch.cuda.current_device());
        generator_AtoB = generator_AtoB.cuda()
        generator_BtoA = generator_BtoA.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    """Train generator and discriminator."""
    total_step = len(data_loader) # For Print Log
    iter = 0
    total_calc_time = [];
    start_time2 = time.time();
    for epoch in range(args.num_epochs):
        start_time = time.time();
        for i, sample in enumerate(data_loader):
            input_A = sample['A']
            input_B = sample['B']

            # ===================== Random Shuffle =====================#
            idx_A = np.arange(input_A.size(0))
            idx_B = np.arange(input_B.size(0))
            np.random.shuffle(idx_A)
            np.random.shuffle(idx_B)

            input_A = input_A.numpy()
            input_B = input_B.numpy()

            input_A = torch.from_numpy(input_A[idx_A])
            input_B = torch.from_numpy(input_B[idx_B])

            A = to_variable(input_A)
            B = to_variable(input_B)

            # ===================== Forward =====================#
            generator_AtoB.zero_grad()
            generator_BtoA.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            A_to_B = generator_AtoB(A)
            B_to_A = generator_BtoA(B)

            A_to_B_to_A = generator_BtoA(A_to_B)
            B_to_A_to_B = generator_AtoB(B_to_A)

            ##########내가한거 내가한거 ################
            A_to_B_to_A_to_B = generator_AtoB(A_to_B_to_A)
            B_to_A_to_B_to_A = generator_BtoA(B_to_A_to_B)
            #############################################


            A_real, A_real_features = discriminator_A(A)
            A_fake, A_fake_features = discriminator_A(B_to_A)

            B_real, B_real_features = discriminator_B(B)
            B_fake, B_fake_features = discriminator_B(A_to_B)

            # ===================== Train D =====================#
            loss_D_A = (GAN_Loss(A_real, True, criterionGAN) + GAN_Loss(A_fake, False, criterionGAN)) * 0.5
            loss_D_B = (GAN_Loss(B_real, True, criterionGAN) + GAN_Loss(B_fake, False, criterionGAN)) * 0.5
            loss_D = loss_D_A + loss_D_B

            # ===================== Train G =====================#
            loss_G_Recon_A = criterionRecon(A_to_B_to_A, A)
            loss_G_Recon_B = criterionRecon(B_to_A_to_B, B)

            loss_G_A = GAN_Loss(A_fake, True, criterionGAN)
            loss_G_B = GAN_Loss(B_fake, True, criterionGAN)

            loss_G_A_feature = Feature_Loss(A_real_features, A_fake_features, criterionFeature)
            loss_G_B_feature = Feature_Loss(B_real_features, B_fake_features, criterionFeature)

            if iter < args.decay_gan_loss:
                rate = args.starting_rate
            else:
                rate = args.changed_rate

            loss_G_A_Total = (loss_G_A*0.1 + loss_G_A_feature*0.9) * (1.-rate) + loss_G_Recon_A * rate
            loss_G_B_Total = (loss_G_B*0.1 + loss_G_B_feature*0.9) * (1.-rate) + loss_G_Recon_B * rate

            loss_G = loss_G_A_Total + loss_G_B_Total

            # ===================== Optimized =====================#

            if iter % 3 == 0:
                loss_D.backward()
                d_optimizer.step()
            else:
                loss_G.backward()
                g_optimizer.step()

            # print the log info
            if (i + 1) % args.log_step == 0:
                print('Iteration [%d], Epoch [%d/%d], BatchStep[%d/%d], D_loss: %.4f, G_loss: %.4f, run time : %s second'
                      % (iter + 1, epoch + 1, args.num_epochs, i + 1, total_step, loss_D.data[0], loss_G.data[0], (time.time()-start_time)))
                #print("run time : %s seconds" %(round(time.time()-start_time, 3)))
            # save the sampled images
            if (iter + 1) % args.sample_step == 0:
                res1 = torch.cat((torch.cat((A, A_to_B), dim=3), A_to_B_to_A), dim=3)
                res2 = torch.cat((torch.cat((B, B_to_A), dim=3), B_to_A_to_B), dim=3)
                res = torch.cat((res1, res2), dim=2)
                torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, 'Generated-%d-%d-%d.png' % (iter + 1, epoch + 1, i + 1)))

            iter += 1
        e = int(time.time() - start_time2)
        total_calc_time.append(str(e))
        print('epoch time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        #print("total epoch time : [%.5d]seconds"%(time.time()-start_time2))
        # save the model parameters for each epoch
        g_pathAtoB = os.path.join(args.model_path, 'generatorAtoB-%d.pkl' % (epoch + 1))
        g_pathBtoA = os.path.join(args.model_path, 'generatorBtoA-%d.pkl' % (epoch + 1))
        torch.save(generator_AtoB.state_dict(), g_pathAtoB)
        torch.save(generator_BtoA.state_dict(), g_pathBtoA)
    for i in range(len(total_calc_time)):
        print("i epoch time : " + total_calc_time[i] )
if __name__ == "__main__":
    main()
