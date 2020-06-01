import os
import traceback
from pathlib import Path

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
from numpy import linalg as npla

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

import math

from facelib.nn_pt import nn as nn_pt

from facelib import FaceType, LandmarksProcessor
from facelib.LandmarksProcessor import convert_98_to_68

from facelib.coord_conv import CoordConvTh


"""
ported from https://github.com/protossw512/AdaptiveWingLoss
"""
class FANExtractor(object):
    def __init__ (self, place_model_on_cpu=False):
        model_path = Path(__file__).parent / "AWL.pth"
        
        nn_pt.initialize()
        
        if not model_path.exists():
            raise Exception("Unable to load AWL.pth")

        def conv3x3(in_planes, out_planes, strd=1, padding=1,
            bias=False,dilation=1):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias,
                     dilation=dilation)

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                # self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                # self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                # out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                # out = self.bn2(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out
        

        class ConvBlock(nn.Module):
            def __init__(self, in_planes, out_planes):
                super(ConvBlock, self).__init__()
                self.in_planes = in_planes
                self.out_planes = out_planes

                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = conv3x3(in_planes, int(out_planes / 2))

                self.bn2 = nn.BatchNorm2d(out_planes//2)
                self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4),
                             padding=1, dilation=1)

                self.bn3 = nn.BatchNorm2d(out_planes//4)
                self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4),
                             padding=1, dilation=1)

                if self.in_planes != self.out_planes:
                    self.downsample = nn.Sequential(
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(True),
                        nn.Conv2d(in_planes, out_planes,
                                  kernel_size=1, stride=1, bias=False),
                    )
                else:
                    self.downsample = None

            def forward(self, x):
                residual = x

                out1 = self.bn1(x)
                out1 = F.relu(out1, True)
                out1 = self.conv1(out1)

                out2 = self.bn2(out1)
                out2 = F.relu(out2, True)
                out2 = self.conv2(out2)

                out3 = self.bn3(out2)
                out3 = F.relu(out3, True)
                out3 = self.conv3(out3)

                out3 = torch.cat((out1, out2, out3), 1)

                if self.downsample is not None:
                    residual = self.downsample(residual)

                out3 += residual

                return out3

        class HourGlass (nn.Module):
            def __init__(self, num_modules, depth, num_features, first_one=False):
                super(HourGlass, self).__init__()
                self.num_modules = num_modules
                self.depth = depth
                self.features = num_features
                self.coordconv = CoordConvTh(x_dim=64, y_dim=64,
                                     with_r=True, with_boundary=True,
                                     in_channels=256, first_one=first_one,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1, padding=0)
                self._generate_network(self.depth)

            def _generate_network(self, level):
                self.add_module('b1_' + str(level), ConvBlock(256, 256))

                self.add_module('b2_' + str(level), ConvBlock(256, 256))

                if level > 1:
                    self._generate_network(level - 1)
                else:
                    self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

                self.add_module('b3_' + str(level), ConvBlock(256, 256))

            def _forward(self, level, inp):
                # Upper branch
                up1 = inp
                up1 = self._modules['b1_' + str(level)](up1)

                # Lower branch
                low1 = F.avg_pool2d(inp, 2, stride=2)
                low1 = self._modules['b2_' + str(level)](low1)

                if level > 1:
                    low2 = self._forward(level - 1, low1)
                else:
                    low2 = low1
                    low2 = self._modules['b2_plus_' + str(level)](low2)

                low3 = low2
                low3 = self._modules['b3_' + str(level)](low3)

                up2 = F.upsample(low3, scale_factor=2, mode='nearest')

                return up1 + up2

            def forward(self, x, heatmap):
                x, last_channel = self.coordconv(x, heatmap)
                return self._forward(self.depth, x), last_channel


        class FAN (nn.Module):
            def __init__(self, num_modules=1, end_relu=False, gray_scale=False, num_landmarks=68):
                super(FAN,self).__init__()
                self.num_modules = num_modules
                self.gray_scale = gray_scale
                self.end_relu = end_relu
                self.num_landmarks = num_landmarks

                # Base part
                if self.gray_scale:
                    self.conv1 = CoordConvTh(x_dim=256, y_dim=256,
                                     with_r=True, with_boundary=False,
                                     in_channels=3, out_channels=64,
                                     kernel_size=7,
                                     stride=2, padding=3)
                else:
                    self.conv1 = CoordConvTh(x_dim=256, y_dim=256,
                                     with_r=True, with_boundary=False,
                                     in_channels=3, out_channels=64,
                                     kernel_size=7,
                                     stride=2, padding=3)
                    
                self.bn1 = nn.BatchNorm2d(64)
                self.conv2 = ConvBlock(64, 128)
                self.conv3 = ConvBlock(128, 128)
                self.conv4 = ConvBlock(128, 256)

                # Stacking part
                for hg_module in range(self.num_modules):
                    if hg_module == 0:
                        first_one = True
                    else:
                        first_one = False
                    self.add_module('m' + str(hg_module), HourGlass(1, 4, 256,
                                                            first_one))
                    self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
                    self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                    self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
                    self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            num_landmarks+1, kernel_size=1, stride=1, padding=0))

                    if hg_module < self.num_modules - 1:
                        self.add_module(
                            'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                        self.add_module('al' + str(hg_module), nn.Conv2d(num_landmarks+1,
                                                                 256, kernel_size=1, stride=1, padding=0))
   


            def forward(self, x):
                x, _ = self.conv1(x)
                x = F.relu(self.bn1(x), True)
                # x = F.relu(self.bn1(self.conv1(x)), True)
                x = F.avg_pool2d(self.conv2(x), 2, stride=2)
                x = self.conv3(x)
                x = self.conv4(x)

                previous = x

                outputs = []
                boundary_channels = []
                tmp_out = None
                for i in range(self.num_modules):
                    hg, boundary_channel = self._modules['m' + str(i)](previous,
                                                               tmp_out)

                    ll = hg
                    ll = self._modules['top_m_' + str(i)](ll)

                    ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)
                    

                    # Predict heatmaps
                    tmp_out = self._modules['l' + str(i)](ll)
                    if self.end_relu:
                        tmp_out = F.relu(tmp_out) # HACK: Added relu

                    outputs.append(tmp_out)
                    boundary_channels.append(boundary_channel)

                    if i < self.num_modules - 1:
                        ll = self._modules['bl' + str(i)](ll)
                        tmp_out_ = self._modules['al' + str(i)](tmp_out)
                        previous = previous + ll + tmp_out_
                        
                return outputs, boundary_channels
        
        def load_model(model, pretrained_path, load_to_cpu):
            if load_to_cpu:
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            else:
                device = torch.cuda.current_device()
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))
                
            if 'state_dict' not in checkpoint:
                model.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint['state_dict']
                model_weights = model.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                              if k in model_weights}
                model_weights.update(pretrained_weights)
                model.load_state_dict(model_weights)
            return model
            
        
        torch.set_grad_enabled(False)
        
        self.model = FAN(num_modules=4, num_landmarks=98, end_relu=False, gray_scale=False)
        self.model = load_model(self.model, model_path, place_model_on_cpu)

        self.model.eval()
        
        self.device = torch.device("cpu" if place_model_on_cpu else "cuda")
        self.model = self.model.to(self.device)


    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape
        print("Image shape:", input_image.shape)

        landmarks = []
        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0

            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
            print("Center:", center)
            centers = [ center ]

            if multi_sample:
                centers += [ center + [-1,-1],
                             center + [1,-1],
                             center + [1,1],
                             center + [-1,1],
                           ]

            images = []
            try:
                for c in centers:
                    images += [ self.crop_old(input_image, c, scale)  ]

                images = np.stack (images)
                images = images.astype(np.float32) / 255.0
                
                i = 0
                for img in images:
                    img = ToTensor()(img)
                    img = img.to(self.device)
                
                    outputs, boundary_channels = self.model(img[None,...])
                
                    pred_heatmap = outputs[-1][:, :-1, :, :][i].detach().cpu()
                
                    pred_landmarks, _ = self.get_pts_from_predict ( pred_heatmap.unsqueeze(0), centers[i], scale)
                    i += 1
                
                    pred_landmarks = pred_landmarks.squeeze().numpy()
                    pred_landmarks = convert_98_to_68(pred_landmarks)
                
                    landmarks += [pred_landmarks]
                
            except:
                landmarks.append (None)

        if second_pass_extractor is not None:
            for i, lmrks in enumerate(landmarks):
                try:
                    if lmrks is not None:
                        image_to_face_mat = LandmarksProcessor.get_transform_mat (lmrks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC )

                        rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                        if len(rects2) == 1: #dont do second pass if faces != 1 detected in cropped image
                            lmrks2 = self.extract (face_image, [ rects2[0] ], is_bgr=is_bgr, multi_sample=True)[0]
                            landmarks[i] = LandmarksProcessor.transform_points (lmrks2, image_to_face_mat, True)
                except:
                    pass

        return landmarks
        
    def transform_old(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def transform(self, point, center, scale, resolution, rotation=0, invert=False):
        _pt = np.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = np.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if rotation != 0:
            rotation = -1*rotation
            r = np.eye(3)
            ang = rotation * math.pi / 180.0
            s = math.sin(ang)
            c = math.cos(ang)
            r[0][0] = c
            r[0][1] = -s
            r[1][0] = s
            r[1][1] = c

            t_ = np.eye(3)
            t_[0][2] = -resolution / 2.0
            t_[1][2] = -resolution / 2.0
            t_inv = torch.eye(3)
            t_inv[0][2] = resolution / 2.0
            t_inv[1][2] = resolution / 2.0
            t = reduce(np.matmul, [t_inv, r, t_, t])

        if invert:
            t = np.linalg.inv(t)
        new_point = (np.matmul(t, _pt))[0:2]

        return new_point.astype(int)
        
    def crop(self, image, center, scale, resolution=256, center_shift=0):
        new_image = cv2.copyMakeBorder(image, center_shift,
                                   center_shift,
                                   center_shift,
                                   center_shift,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
        if center_shift != 0:
            center[0] += center_shift
            center[1] += center_shift
        length = 200 * scale
        top = int(center[1] - length // 2)
        bottom = int(center[1] + length // 2)
        left = int(center[0] - length // 2)
        right = int(center[0] + length // 2)
        y_pad = abs(min(top, new_image.shape[0] - bottom, 0))
        x_pad = abs(min(left, new_image.shape[1] - right, 0))
        top, bottom, left, right = top + y_pad, bottom + y_pad, left + x_pad, right + x_pad
        new_image = cv2.copyMakeBorder(new_image, y_pad,
                                   y_pad,
                                   x_pad,
                                   x_pad,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
        new_image = new_image[top:bottom, left:right]
        new_image = cv2.resize(new_image, dsize=(int(resolution), int(resolution)),
                           interpolation=cv2.INTER_LINEAR)
        return new_image

    def crop_old(self, image, center, scale, resolution=256.0):
        ul = self.transform_old([1, 1], center, scale, resolution).astype( np.int )
        br = self.transform_old([resolution, resolution], center, scale, resolution).astype( np.int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg
    

    def get_pts_from_predict(self, hm, center=None, scale=None, rot=None):
        max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
        preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                    preds[i, j].add_(diff.sign_().mul_(.25))

        preds.add_(-0.5)

        preds_orig = torch.zeros(preds.size())
        if center is not None and scale is not None:
            for i in range(hm.size(0)):
                for j in range(hm.size(1)):
                    preds_orig[i, j] = torch.from_numpy(self.transform(preds[i, j], center, scale, hm.size(2), rot if (rot is not None) else (0), True))

        return preds, preds_orig
