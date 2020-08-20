# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.pisnet import pisnet, BasicBlock, Bottleneck

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class PISNet(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, has_non_local="no", sia_reg="no", pyramid="no", test_pair="no"):
        super(PISNet, self).__init__()

        self.base = pisnet(last_stride=last_stride,
                           block=Bottleneck,
                           layers=[3, 4, 6, 3], has_non_local=has_non_local, sia_reg=sia_reg, pyramid=pyramid)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.test_pair = test_pair
        self.sia_reg = sia_reg

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x_g, x, x_g2=[], is_first=False):

        feature_gallery, gallery_attention, feature_gallery1, gallery_attention1, feature_query, reg_feature_query, reg_query_attention = self.base(x_g, x, x_g2=x_g2, is_first=is_first)

        global_feat = self.gap(feature_gallery)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        # gallery_attention = gallery_attention.view(gallery_attention.shape[0], -1)

        if self.training:
            global_feat1 = self.gap(feature_gallery1)
            global_feat1 = global_feat.view(global_feat1.shape[0], -1)
            gallery_attention1 = gallery_attention.view(gallery_attention1.shape[0], -1)

            global_feature_query = self.gap(feature_query)
            global_feature_query = global_feat.view(global_feature_query.shape[0], -1)

            if self.sia_reg == "yes":
                global_reg_query = self.gap(reg_feature_query)
                global_reg_query = global_feat.view(global_reg_query.shape[0], -1)
                reg_query_attention = gallery_attention.view(reg_query_attention.shape[0], -1)

        # cls_score_pos_neg = self.classifier_attention(gallery_attention)
        # cls_score_pos_neg = self.sigmoid(cls_score_pos_neg)

        if self.neck == 'no':
            feat = global_feat
            if self.training:
                feat1 = global_feat1
                if self.sia_reg == "yes":
                    feat2 = global_reg_query
            # feat_query = global_feature_query

            # feat_guiding = global_feat_guiding
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            if self.training:
                feat1 = self.bottleneck(global_feat1)  # normalize for angular softmax
                if self.sia_reg == "yes":
                    feat2 = self.bottleneck(global_reg_query)
                # feat_query = self.bottleneck(global_feature_query)

                # feat_guiding = self.bottleneck(global_feat_guiding)
        if self.training:
            cls_score = self.classifier(feat)
            cls_score1 = self.classifier(feat1)
            cls_score2 = self.classifier(feat2)
            # cls_score_guiding = self.classifier(feat_guiding)
            return cls_score, global_feat, cls_score1, global_feat1, global_feature_query, cls_score2, global_reg_query   # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])







