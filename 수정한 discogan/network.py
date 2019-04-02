import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, extra_layers=False):
        super(Generator, self).__init__()

        if extra_layers == False:
            # For Edges/Shoes/Handbags and Facescrub

                # [-1, 3, 64x64] -> [-1, 64, 32x32]
                self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                )
                # [-1, 128, 16x16]
                self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                )
                # [-1, 256, 8x8]
                self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                )
                # [-1, 512, 4x4]
                self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                )
                # [-1, 256, 8x8]
                self.conv5 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                )
                # [-1, 128, 16x16]
                self.conv6 =nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                #nn.ReLU(True),
                )
                # [-1, 256, 32x32]
                self.conv7 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                #nn.ReLU(True),
                )
                # [-1, 3, 64x64]
                self.conv8 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Tanh()
                )
                self.test = nn.Sequential(
                nn.ConvTranspose2d(512,128,6,4,1,bias=False),
                nn.BatchNorm2d(128),
                )
                self.test2 = nn.Sequential(
                nn.ConvTranspose2d(256,64,6,4,1,bias=False),
                nn.BatchNorm2d(64),
                )
                self.add1 = nn.Sequential(
                nn.ReLU(True),
                )




    def forward(self, input):
        output1 = self.conv1(input)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        a1 = self.test(output4)
        output5 = self.conv5(output4)#+ output3
        a2 = self.test2(output5)
        output6 = self.conv6(output5)
        resi = self.add1(output6+a1)#+ ResidualBlock(128) #+ output2
        output7 = self.conv7(resi)#resi)
        resi2 = self.add1(output7+a2) #+ ResidualBlock(64)
        out = self.conv8(resi2)#resi2)
        #output9 = self.conv9(output8)
        #output10 = self.conv10(output9)
        #output11 = self.conv11(output10)
        #output12 = self.conv12(output11)
        #output13 = self.conv13(output12)+output11
        #output14 = self.conv14(output13)+output10
        #output15 = self.conv15(output14)
        #out = self.conv16(output15)

        return out
class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        # [-1, 3, 64x64] -> [-1, 64, 32x32]
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.layer1 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 128, 16x16]
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 256, 8x8]
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 512, 4x4]
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 1, 1x1]
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        layer1 = self.layer1( self.conv1( input ) )
        layer2 = self.layer2( self.bn2( self.conv2( layer1 ) ) )
        layer3 = self.layer3( self.bn3( self.conv3( layer2 ) ) )
        layer4 = self.layer4( self.bn4( self.conv4( layer3 ) ) )

        ##layer5 = self.conv5(layer4)
        sigmoid = nn.Sigmoid()
        feature = [layer2, layer3, layer4]

        return sigmoid(self.conv5(layer4)), feature
