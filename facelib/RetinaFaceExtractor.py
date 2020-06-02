import operator
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from facelib.nn_pt import nn as nn_pt

import torchvision.models as models
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
from torchvision.transforms import ToTensor

from facelib.net import FPN as FPN
from facelib.net import SSH as SSH

from facelib.box_utils import decode
from facelib.prior_box import PriorBox

from facelib.ResNet50 import resnet50

from facelib.config import cfg_re50
"""Ported from https://github.com/biubug6/Pytorch_Retinaface/"""

class RetinaFaceExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        nn_pt.initialize()
        
        model_path = Path(__file__).parent / "RetinaFace-Resnet50.pth"
        
        if not model_path.exists():
            raise Exception("Unable to load RetinaFace-Resnet50.pth")

        class ClassHead(nn.Module):
            def __init__(self,inchannels=512,num_anchors=3):
                super(ClassHead,self).__init__()
                self.num_anchors = num_anchors
                self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

            def forward(self,x):
                out = self.conv1x1(x)
                out = out.permute(0,2,3,1).contiguous()
        
                return out.view(out.shape[0], -1, 2)
            

        class BboxHead(nn.Module):
            def __init__(self,inchannels=512,num_anchors=3):
                super(BboxHead,self).__init__()
                self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

            def forward(self,x):
                out = self.conv1x1(x)
                out = out.permute(0,2,3,1).contiguous()

                return out.view(out.shape[0], -1, 4)

            
        class LandmarkHead(nn.Module):
            def __init__(self,inchannels=512,num_anchors=3):
                super(LandmarkHead,self).__init__()
                self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

            def forward(self,x):
                out = self.conv1x1(x)
                out = out.permute(0,2,3,1).contiguous()

                return out.view(out.shape[0], -1, 10)

        class RetinaFace(nn.Module):
            def __init__(self, cfg = cfg_re50):
                super(RetinaFace,self).__init__()
                backbone = resnet50(pretrained=cfg['pretrain'])

                self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
                in_channels_stage2 = cfg['in_channel']
                in_channels_list = [in_channels_stage2 * 2,
                                    in_channels_stage2 * 4,
                                    in_channels_stage2 * 8,]
                out_channels = cfg['out_channel']

                self.fpn = FPN(in_channels_list,out_channels)
                self.ssh1 = SSH(out_channels, out_channels)
                self.ssh2 = SSH(out_channels, out_channels)
                self.ssh3 = SSH(out_channels, out_channels)

                self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
                self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
                self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

            def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
                classhead = nn.ModuleList()
                for i in range(fpn_num):
                    classhead.append(ClassHead(inchannels,anchor_num))
                return classhead

            def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
                bboxhead = nn.ModuleList()
                for i in range(fpn_num):
                    bboxhead.append(BboxHead(inchannels,anchor_num))
                return bboxhead

            def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
                landmarkhead = nn.ModuleList()
                for i in range(fpn_num):
                    landmarkhead.append(LandmarkHead(inchannels,anchor_num))
                return landmarkhead


            def forward(self,inputs):
                out = self.body(inputs)

                # FPN
                fpn = self.fpn(out)

                # SSH
                feature1 = self.ssh1(fpn[0])
                feature2 = self.ssh2(fpn[1])
                feature3 = self.ssh3(fpn[2])
                features = [feature1, feature2, feature3]

                bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
                classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
                ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

                output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

                return output
                
        def check_keys(model, pretrained_state_dict):
            ckpt_keys = set(pretrained_state_dict.keys())
            model_keys = set(model.state_dict().keys())
            used_pretrained_keys = model_keys & ckpt_keys
            unused_pretrained_keys = ckpt_keys - model_keys
            missing_keys = model_keys - ckpt_keys
            assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
            return True
                
        def remove_prefix(state_dict, prefix):
            ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}
            
        def load_model(model, pretrained_path, load_to_cpu):
            if load_to_cpu:
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            else:
                device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            check_keys(model, pretrained_dict)
            model.load_state_dict(pretrained_dict, strict=False)
            return model
        
        try:
            torch.set_grad_enabled(False)
            self.model = RetinaFace(cfg=cfg_re50)
            self.model = load_model(self.model, model_path, place_model_on_cpu)
            self.model.eval()
        
            self.device = torch.device("cpu" if place_model_on_cpu else "cuda")
            self.model = self.model.to(self.device)
            
        except:
            self.model = None
            print("Could not load RetinaFace")
        
        

        

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, is_bgr=True, is_remove_intersects=False):
        
        cfg = cfg_re50
        
        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize (input_image, ( int(w/input_scale), int(h/input_scale) ), interpolation=cv2.INTER_LINEAR)

        (h, w, ch) = input_image.shape
        
        with torch.no_grad():
            
            input_image = ToTensor()(input_image)
            input_image = input_image.to(self.device)
        
            loc, conf, landmarks = self.model( input_image[None,...]  )
        
            priorbox = PriorBox(cfg, image_size=(h, w))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
        
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = np.float32(boxes.cpu().numpy())
            scores = np.float32(conf.squeeze(0).data.cpu().numpy()[:, 1])
        
            inds = np.where(scores > 0.05)[0]
            boxes = boxes[inds]
            scores = scores[inds]
        
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            
        keep = self.refine_nms(dets, 0.3)
        dets = dets[keep, :]
        
        dets = dets.tolist()
        dets = [ x[:-1].astype(np.int) for x in dets if x[-1] >= 0.5 ]
        
        

        detected_faces = []
        
        for ltrb in dets:
            # l,t,r,b = [ x for x in ltrb]
            l,t,r,b = [ x*input_scale for x in ltrb]
            bt = b-t
            if min(r-l,bt) < 40: #filtering faces < 40pix by any side
                continue
            b += bt*0.1 #enlarging bottom line a bit for 2DFAN-4, because default is not enough covering a chin
            detected_faces.append ( [int(x) for x in (l,t,r,b) ] )

        #sort by largest area first
        detected_faces = [ [(l,t,r,b), (r-l)*(b-t) ]  for (l,t,r,b) in detected_faces ]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True )
        detected_faces = [ x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1 = detected_faces[i]
                l0,t0,r0,b0 = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)

        return detected_faces 
            
        

    def refine(self, olist):
        bboxlist = []
        for i, ((ocls,), (oreg,)) in enumerate (  ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls[...,1] > 0.05)):
                score = ocls[hindex, windex, 1]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        bboxlist = [ x[:-1].astype(np.int) for x in bboxlist if x[-1] >= 0.5]
        return bboxlist
        
    def refine_nms2(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
        

    def refine_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
