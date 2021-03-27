import numpy as np
import torch as pt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -0.05, 0.05)
        m.bias.data.fill_(0.001)
    elif type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight, -0.05, 0.05)
        m.bias.data.fill_(0.001)

class Func(nn.Module):
    def __init__(self, functional):
        nn.Module.__init__(self)
        self.functional = functional

    def forward(self, *input):
        return self.functional(*input)


class FeatureExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.pool4 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(128,128,3,padding=1)
        self.conv6 = nn.Conv2d(128,128,3,padding=1)
        self.pool6 = nn.MaxPool2d(2,2)
        self.conv7 = nn.Conv2d(128,256,3,padding=1)
        self.conv8 = nn.Conv2d(256,256,3,padding=1)
        self.pool8 = nn.MaxPool2d(2,2)
        self.conv9 = nn.Conv2d(256,256,3,padding=1)
        self.conv10 = nn.Conv2d(256,512,3,padding=1)
        self.pool10 = nn.MaxPool2d(2,2)
        self.conv11 = nn.Conv2d(512,512,3,padding=1)
        self.apply(init_weights)

    def forward(self, input):
        """
        if the input
        """
        #print("\tIn Model: input size", input.size()) 
        #conv1 -> relu -> conv2 -> relu -> pool2 -> conv3 -> relu
        x3 = F.relu(self.conv3(self.pool2(F.relu(self.conv2(F.relu(self.conv1(input)))))))
        # conv4 -> relu -> pool4 -> conv5 -> relu
        x5 = F.relu(self.conv5(self.pool4(F.relu(self.conv4(x3)))))
        # conv6 -> relu -> pool6 -> conv7 -> relu
        x7 = F.relu(self.conv7(self.pool6(F.relu(self.conv6(x5)))))
        # conv8 -> relu -> pool8 -> conv9 -> relu
        x9 = F.relu(self.conv9(self.pool8(F.relu(self.conv8(x7)))))
        # conv10 -> relu -> pool10 -> conv11 -> relU
        x11 = F.relu(self.conv11(self.pool10(F.relu(self.conv10(x9)))))
        return pt.cat((
            self.flatten(x3),
            self.flatten(x5),
            self.flatten(x7),
            self.flatten(x9),
            self.flatten(x11),
        ), 1), x11.view(x11.size(0), -1)

    def flatten(self, x):
        """
        change vector from BxCxHxW to BxCHW
        """
        B, C, H, W = x.size()
        return x.view(B, C*H*W)

class Comparitor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fcs = nn.Sequential(
            nn.Linear(120832, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            Func(lambda x: x*1e-2),
            nn.Linear(1, 1))
        self.weights = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            Func(lambda x: x+1e-6))

    def forward(self, featureA, coarseA, featureRef, coarseRef):
        scores = self.fcs(featureRef - featureA)
        weights = self.weights(coarseRef - coarseA)
        final_score = (scores * weights).sum() / weights.sum()
        return final_score, scores, weights



class DimTransformer(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Patch height, patch width
        self.pH, self.pW = 64, 64


    def calc_numb_patches(self,H,W,pH,S):
        return int(((H-pH+S)/S)*((W-pH+S)/S))


    def forward(self, images,stride):

        ImBS, ImC, ImH, ImW =  images.shape
        # Patch Batch Size

        #PBS = int(ImBS*self.calc_numb_patches(ImH,ImW,self.pH,stride))
        patches = images.unfold(2, self.pH, stride).unfold(3, self.pW, stride)
        patches = pt.transpose(patches,1,3)

        PBS = int(pt.numel(patches)/ImC/self.pH/self.pW)
        patches = patches.reshape(PBS,ImC,self.pH,self.pW)
        return patches,PBS,ImBS



class PUTransformer(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.P_min =0.3176
        self.P_max =1270.1545
        self.logL_min = -1.7647
        self.logL_max = 8.9979
        self.sixth_order = pt.tensor([-2.76770417e-03,  5.45782779e-02, -1.31306184e-01, -4.05827702e+00, 3.74004810e+01,  2.84345651e+01,  5.15162034e+01])
        self.third_order = pt.tensor([2.5577829 , 17.73608751, 48.96952155, 45.55950728])
        self.epsilon = 1e-8
        
    def forward(self, im, im_type='sdr', lum_top=100, lum_bottom=0.5):
        
        im = self.apply_disp_model( im,im_type, lum_top=100, lum_bottom=0.5)
        im = self.clamp_image(im)
        im = self.apply_pu(im)
        im = self.scale(im)
        return im
    
    def apply_disp_model(self, im, im_type, lum_top=100.0, lum_bottom=0.5):
        if im_type == 'hdr':
            return im
        else:
            return (lum_top - lum_bottom) * ((im/255.0)**2.2) + lum_bottom
        

    def clamp_image(self,im):
        epsilon = 1e-8
        return pt.clamp(pt.log10(pt.clamp(im, epsilon, None)), self.logL_min,self.logL_max)
        
    def apply_pu(self, img):
        third_ord = self.third_order[0]*img**3+self.third_order[1]*img**2+self.third_order[2]*img+self.third_order[3]
        sixth_ord = self.sixth_order[0]*img**6+self.sixth_order[1]*img**5+self.sixth_order[2]*img**4+self.sixth_order[3]*img**3+self.sixth_order[4]*img**2+self.sixth_order[5]*img+self.sixth_order[6]

        return (img>=0.8).int()*sixth_ord+(img<0.8).int()*third_ord

    def scale(self, x):
        """
        scale x to values between 0 and 1
        """

        return (x - self.P_min ) / (self.P_max - self.P_min)