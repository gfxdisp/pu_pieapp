import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from models.simple import FeatureExtractor, Comparitor, PUTransformer,DimTransformer


class PUPieAPP(nn.Module):
    
    def __init__(self,state=None):
        nn.Module.__init__(self)
        self.pu_transformer = PUTransformer()
        self.dim_transformer = DimTransformer()
        self.extractor = FeatureExtractor()
        self.comparitor = Comparitor()

        if state:
            self.extractor.load_state_dict(state['extractor'])
            self.comparitor.load_state_dict(state['comparitor'])

    def forward(self, img, ref, im_type='sdr', lum_top=100, lum_bottom=0.5, stride=64):
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

class PUPieAPPPatch(nn.Module):
    
    def __init__(self, state=None):
        nn.Module.__init__(self)
        self.pu_transformer = PUTransformer()
        self.extractor = FeatureExtractor()
        self.comparitor = Comparitor()

        if state:
            self.extractor.load_state_dict(state['extractor'])
            self.comparitor.load_state_dict(state['comparitor'])

    def forward(self, img, ref, im_type='sdr', lum_top=100, lum_bottom=0.5):
        img = self.pu_transformer(img, im_type,  lum_top, lum_bottom)
        ref = self.pu_transformer(ref, im_type, lum_top, lum_bottom)
        f1, c1 = self.extractor(img)
        f2, c2 = self.extractor(ref)
        f_score, scores, weights = self.comparitor(f1, c1, f2, c2)
        return f_score,scores,weights
