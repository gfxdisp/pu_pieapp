import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class PerceptLossNet(nn.Module):
    def __init__(self,pu_transformer, extract_net):
        nn.Module.__init__(self)
        self.pu_transformer = pu_transformer
        self.extractor = extract_net

    def forward(self, img, im_type='ldr', lum_top=100, lum_bottom=0.5):
        img = self.pu_transformer(img, im_type,  lum_top, lum_bottom)
        x3, x5, x7, x9, x11 = self.extractor(img)
        return x3, x5, x7, x9, x11



class PUPieAppEndToEnd(nn.Module):
    
    def __init__(self, pu_transformer, dim_transformer, extract_net, compare_net):
        nn.Module.__init__(self)
        self.pu_transformer = pu_transformer
        self.dim_transformer = dim_transformer
        self.extractor = extract_net
        self.comparitor = compare_net

    def forward(self, img, ref, im_type='ldr', lum_top=100, lum_bottom=0.5, stride=64):
        img = self.pu_transformer(img, im_type,  lum_top, lum_bottom)
        ref = self.pu_transformer(ref, im_type, lum_top, lum_bottom)

        img,PBS,ImBS = self.dim_transformer(img, stride)
        ref,PBS,ImBS = self.dim_transformer(ref, stride)
        f1, c1 = self.extractor(img)
        f2, c2 = self.extractor(ref)


        _, scores, weights = self.comparitor(f1, c1, f2, c2)

        per_patch_score = scores.view(ImBS,1,int(PBS/ImBS))
        per_patch_weight = weights.view(ImBS,1,int(PBS/ImBS))
        score = (per_patch_score*per_patch_weight).sum(2)/((per_patch_weight).sum(2))

        return score


class PUPieApp(nn.Module):
    
    def __init__(self, pu_transformer, extract_net, compare_net):
        nn.Module.__init__(self)
        self.pu_transformer = pu_transformer
        self.extractor = extract_net
        self.comparitor = compare_net

    def forward(self, img, ref, im_type='ldr', lum_top=100, lum_bottom=0.5):
        img = self.pu_transformer(img, im_type,  lum_top, lum_bottom)
        ref = self.pu_transformer(ref, im_type, lum_top, lum_bottom)
        f1, c1 = self.extractor(img)
        f2, c2 = self.extractor(ref)
        f_score, scores, weights = self.comparitor(f1, c1, f2, c2)
        return f_score,scores,weights

class ScoreNet(nn.Module):
    def __init__(self, extract_net, compare_net):
        nn.Module.__init__(self)
        self.extractor = extract_net
        self.comparitor = compare_net

    def forward(self, img, ref):
        f1, c1 = self.extractor(img)
        f2, c2 = self.extractor(ref)
        f_score, _, _ = self.comparitor(f1, c1, f2, c2)
        return f_score

class ScoreNetScoresWeights(nn.Module):
    def __init__(self, extract_net, compare_net):
        nn.Module.__init__(self)
        self.extractor = extract_net
        self.comparitor = compare_net

    def forward(self, img, ref):
        f1, c1 = self.extractor(img)
        f2, c2 = self.extractor(ref)
        f_score, scores, weights = self.comparitor(f1, c1, f2, c2)
        return f_score,scores,weights

