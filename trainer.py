from __future__ import  absolute_import
import os
from collections import namedtuple
import time
import torch
import torch.nn
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from domain_transfer_network.train import extract_features
from domain_transfer_network.train import gram
from torch import nn
import torch as t
import numpy as np
from utils import array_tool as at
from utils.vis_tool import Visualizer
from Da_module.global_classifier import GRLayer,grad_reverse,_ImageDA
import torchvision
from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter


#-------------------21.11.5-------------------------------
# 预留地，这个地方应该要加上域自适应分类器的loss
#----------------------------------------------------------

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'global_da_loss_low',
                        'global_da_loss_mid',
                        'global_da_loss',
                        'total_loss'
                        ])

#-------------------21.11.12-------------------------------
LossTupleTarget=namedtuple('LossTuple',
                       [
                        'global_da_loss_low',
                        'global_da_loss_mid',
                        'global_da_loss',
                        'total_loss'
                        ])
#----------------------------------------------------------

class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        # -------------------21.11.5------------添加全局自适应判别器模型，vgg9获得深度权重,读取平均格拉姆矩阵
        self.global_da_classifier_low = _ImageDA(256)
        self.global_da_classifier_low.train()
        self.global_da_classifier_mid = _ImageDA(512)
        self.global_da_classifier_mid.train()
        self.global_da_classifier = _ImageDA(512)
        self.global_da_classifier.train()
        self.weight_network = torchvision.models.__dict__['vgg16'](pretrained=True).features.cuda()
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.mean_gram_low = torch.from_numpy(np.load('./mean_gram/vgg16_mean_gram1.npy')).cuda()
        self.mean_gram_mid = torch.from_numpy(np.load('./mean_gram/vgg16_mean_gram2.npy')).cuda()
        self.mean_gram = torch.from_numpy(np.load('./mean_gram/vgg16_mean_gram3.npy')).cuda()
        # ----------------------------------------------------------

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(9)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    # -------------------21.11.5-------------------------------训练时多加一个参数 s_or_t 表示图片来自目标域还是原域
    def forward(self, imgs, bboxes, labels, scale,img_target,ii):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses



        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # features = self.faster_rcnn.extractor(imgs)

        # ----------------------------------------------------------提取三层特征
        low_features, mid_features, features = extract_features(self.faster_rcnn.extractor,
                                                                imgs,
                                                                [15, 22, 29])
        low_features_t, mid_features_t, features_t = extract_features(self.faster_rcnn.extractor,
                                                                      img_target,
                                                                      [15, 22, 29])

        low_weight, mid_weight, weight = extract_features(self.weight_network,
                                                          imgs,
                                                          [15, 22, 29])
        low_weight_t, mid_weight_t, weight_t = extract_features(self.weight_network,
                                                                img_target,
                                                                [15, 22, 29])

        low_weight = self.mse_criterion(gram(low_weight), self.mean_gram_low).detach()
        mid_weight = self.mse_criterion(gram(mid_weight), self.mean_gram_mid).detach()
        weight = self.mse_criterion(gram(weight), self.mean_gram).detach()
        weight_all = torch.softmax(torch.Tensor([low_weight, mid_weight, weight]), dim=0)

        low_weight_t = self.mse_criterion(gram(low_weight_t), self.mean_gram_low).detach()
        mid_weight_t = self.mse_criterion(gram(mid_weight_t), self.mean_gram_mid).detach()
        weight_t = self.mse_criterion(gram(weight_t), self.mean_gram).detach()
        weight_all_t = torch.softmax(torch.Tensor([low_weight_t, mid_weight_t, weight_t]), dim=0)

        # -------------------21.11.5-------------------------------这里增加提取多制度的特征features_1,features_2
        # features_1=self.faster_rcnn.extractor_1(imgs)
        # features_2=self.faster_rcnn.extractor_2(imgs)
        # features_3 = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor, rpn_cls_map = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # ------------提取目标域特征
        _, _, _, _, _, rpn_cls_map_t = \
            self.faster_rcnn.rpn(features_t, img_size, scale)
        # ---------------------------------------------------------------仿照学姐计算mask

        mask = torch.sum(rpn_cls_map[:, 9:, :, :], dim=1)
        mask[mask > 1.0] = 1.0
        mask = (mask + 1.0) / 2
        mask = mask.unsqueeze(0).detach()
        mask_low = F.interpolate(mask, size=(low_features.shape[2], low_features.shape[3])).detach()
        mask_mid = F.interpolate(mask, size=(mid_features.shape[2], mid_features.shape[3])).detach()

        mask_t = torch.sum(rpn_cls_map_t[:, 9:, :, :], dim=1)
        mask_t[mask_t > 1.0] = 1.0
        mask_t = (mask_t + 1.0) / 2
        mask_t = mask_t.unsqueeze(0).detach()
        mask_low_t = F.interpolate(mask_t, size=(low_features_t.shape[2], low_features_t.shape[3])).detach()
        mask_mid_t = F.interpolate(mask_t, size=(mid_features_t.shape[2], mid_features_t.shape[3])).detach()
        # -------------------21.11.5-------------------------------
        # 预留地，这部分有应该将三个提取到的特征取样到相同尺度，然后进行通道拼接，然后输入到分类器，得到一个loss
        # 将输出前两层的张量池化到相同尺度
        # pool1 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        # pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # features_1 = pool1(features_1)
        # features_2 = pool2(features_2)

        # 将三层的特征按照通道拼接成一个大张量，输入到分类器中
        # feature_connect=torch.cat([features_1,features_2,features_3],1)

        s_or_t_pre_low = self.global_da_classifier_low(low_features * mask_low)
        s_or_t_pre_mid = self.global_da_classifier_mid(mid_features * mask_mid)
        s_or_t_pre = self.global_da_classifier(features * mask)

        s_or_t_pre_low_t = self.global_da_classifier_low(low_features_t * mask_low_t)
        s_or_t_pre_mid_t = self.global_da_classifier_mid(mid_features_t * mask_mid_t)
        s_or_t_pre_t = self.global_da_classifier(features_t * mask_t)
        # ----------------------------------------------------------

        # 正向传播有效的其实只有rois
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)
        # print(roi_cls_loc.shape,roi_score.shape)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        # print(gt_rpn_loc.shape,gt_rpn_label.shape)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        # print(gt_roi_label)
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        # -------------------21.11.5-------------------------------
        # 预留地，这里应该是要将域自适应的loss加上去
        # criterion_da = nn.CrossEntropyLoss()
        # s_or_t_pre=s_or_t_pre.squeeze(dim=2)
        # if s_or_t==('s',):
        #     self.s_or_t_gt=torch.Tensor([0]).long().cuda()
        # else:
        #     self.s_or_t_gt=torch.Tensor([1]).long().cuda()
        # global_da_loss=criterion_da(s_or_t_pre,self.s_or_t_gt)
        # global_da_loss*=0.001
        # global_da_loss*=torch.Tensor(0.001).float().cuda()

        # 全局域自适应，邱斌的版本

        global_da_loss_low = -torch.log(s_or_t_pre_low + 1e-9)
        global_da_loss_low_t = -torch.log(1 - s_or_t_pre_low_t + 1e-9)

        global_da_loss_mid = -torch.log(s_or_t_pre_mid + 1e-9)
        global_da_loss_mid_t = -torch.log(1 - s_or_t_pre_mid_t + 1e-9)

        global_da_loss = -torch.log(s_or_t_pre + 1e-9)
        global_da_loss_t = -torch.log(1 - s_or_t_pre_t + 1e-9)

        global_da_loss_low = (global_da_loss_low.mean() * weight_all[0] +
                              global_da_loss_low_t.mean() * weight_all_t[0]) / 2

        global_da_loss_mid = (global_da_loss_mid.mean() * weight_all[1] +
                              global_da_loss_mid_t.mean() * weight_all_t[1]) / 2
        global_da_loss = (global_da_loss.mean() * weight_all[2] +
                          global_da_loss_t.mean() * weight_all[2]) / 2
        # ----------------------------------------------------------

        # -------------------21.11.12----------图像来自目标域时只算da损失----------
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss,
                  global_da_loss_low, global_da_loss_mid, global_da_loss]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)

        # ----------------------------------------------------------

    # -------------------21.11.5-------------------------------训练时多加一个参数 s_or_t 表示图片来自目标域还是原域
    def train_step(self, imgs, bboxes, labels, scale, img_target,ii):
        self.optimizer.zero_grad()

        losses = self.forward(imgs, bboxes, labels, scale, img_target,ii)

        # -------------------21.11.12---------当来图像来自目标域时只更新da损失-------------

        losses.total_loss.backward()
        self.update_meters(losses)
        # -----------------------------------------------------------------------------
        self.optimizer.step()
        # self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            #print("key:",loss_d[key])
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        # for k, v in self.meters.items():
        #     print(v.value()[0])
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    # print("in_weight:",in_weight.shape)
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
