import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np
from functools import partial

class stagenet_references_vis(nn.Module):
    def __init__(self, inverse_depth=False, attn_temp=2, use_visi_net=True, vis_up=True, vis_use_ref=True):
        super(stagenet_references_vis, self).__init__()
        self.inverse_depth = inverse_depth
        self.use_visi_net = use_visi_net
        self.vis_up = vis_up
        self.vis_use_ref = vis_use_ref
        self.tmp = [10,10,10,1]

    def forward(self, ref_features, src_features, proj_matrices, depth_hypo, regnet, stage_idx, vis_net=None, group_cor=False,
                group_cor_dim=8, split_itv=1, up_vis_weights=[]):

        # step 1. feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_features[0].shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0

        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        # print(len(ref_features))
        # step 2. Epipolar Transformer Aggregation
        cor_weights = []
        cor_weights_= []
        for src_idx, (ref_feature, src_fea, src_proj) in enumerate(zip(ref_features, src_features, src_projs)):
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W
            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src) ** 2  # B C D H W
            del warped_src, src_proj, src_fea
            if self.use_visi_net:
                # print("test")
                cor_feat_fuse = cor_feat.reshape(B,group_cor_dim*D,H,W)

                if self.vis_up:
                    if stage_idx == 0:
                        if self.vis_use_ref:
                            cor_weight = vis_net(ref_feature, cor_feat_fuse) # B H W
                        else:
                            cor_weight = vis_net(cor_feat_fuse)  # B H W
                    else:
                        up_vis_weight = up_vis_weights[src_idx]
                        # print(up_vis_weight.shape)
                        up_vis_weight = F.interpolate(up_vis_weight.unsqueeze(1), [cor_feat_fuse.shape[2], cor_feat_fuse.shape[3]], mode="bilinear")
                        if self.vis_use_ref:
                            cor_weight = vis_net(ref_feature, cor_feat_fuse, up_vis=up_vis_weight)
                        else:
                            cor_weight = vis_net(cor_feat_fuse, up_vis=up_vis_weight)
                else:
                    if self.vis_use_ref:
                        cor_weight = vis_net(ref_feature, cor_feat_fuse)  # B H W
                    else:
                        cor_weight = vis_net(cor_feat_fuse)  # B H W

                cor_weights.append(cor_weight)
                cor_weight_ = torch.where(cor_weight>0.05, cor_weight, 0)
                cor_weights_.append(cor_weight)
                # print("cor_weights_", len(cor_weights_))
                cor_weight_sum += cor_weight  # B H W
                # cor_weight_sum += cor_weight_  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
                # cor_feats += cor_weight_.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
                del cor_feat
            else:

                cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                # cor_weight_ = cor_weight
                cor_weights.append(cor_weight)
                cor_weight_ = torch.where(cor_weight > 0.05, cor_weight, 0)
                cor_weights_.append(cor_weight)
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W

                del cor_weight, cor_feat, cor_weight_
        if self.use_visi_net:
            cor_feats = cor_feats / (cor_weight_sum.unsqueeze(1).unsqueeze(1) + 1e-7) # B C D H W
        else:
            # if not self.attn_fuse_d:
            cor_feats = cor_feats / (cor_weight_sum.unsqueeze(1).unsqueeze(1) + 1e-7)  # B C D H W


        del cor_weight_sum, src_features

        # step 3. regularization
        attn_weight_ = regnet(cor_feats)  # B D H W
        del cor_feats

        attn_weight = F.softmax(attn_weight_, dim=1)  # B D H W

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W
        if self.training:
            depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W
        else:
            # print("reg")
            depth = depth_regression(F.softmax(attn_weight_ * self.tmp[stage_idx], dim=1), depth_values=depth_hypo)

        if not self.training:
            with torch.no_grad():
                photometric_confidence = attn_weight.max(1)[0]  # B H W
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),
                                                       scale_factor=2 ** (3 - stage_idx), mode='bilinear',
                                                       align_corners=True).squeeze(1)
        else:
            photometric_confidence = torch.tensor(0.0, dtype=torch.float32, device=ref_feature.device,
                                                  requires_grad=False)
        # print(len(cor_weights_), len(cor_weights))
        ret_dict = {"depth": depth, "photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo,
                    "attn_weight": attn_weight, "cor_weights":cor_weights, "cor_weights_":cor_weights_}

        if self.inverse_depth:
            last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
            inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth

        return ret_dict

def homo_warping_base(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy


    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def homo_warping_3D_with_mask_self_grid(src_fea, src_proj, ref_proj, depth_values, self_line_grid):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    C = src_fea.shape[1]
    Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        xyz = self_line_grid
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea

def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                requires_grad=False).reshape(1, -1) * new_interval.squeeze(1)) #(B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) #(B, D, H, W)
    return depth_range_samples

def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:,None, None, None] + (inverse_depth_min - inverse_depth_max)[:,None, None, None] * itv

    return 1./inverse_depth_hypo

def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    #cur_depth_min, (B, H, W)
    #cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H//2, W//2)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_inverse_range_depth_up(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    #cur_depth_min, (B, H, W)
    #cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    # inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:,None,None])  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:,None,None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, gn=False, group_channel=4):
        super(ConvBnReLU3D, self).__init__()
        if gn == 'IN':
            bn = 'IN'
        else:
            bn = not gn
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        if bn == 'IN':
            self.bn = nn.InstanceNorm3d(out_channels)
        elif bn:
            # print("ConvBnReLU3D bn")
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            # print("ConvBnReLU3D gn")
            self.bn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D_CAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_CAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D*H*W).mean(2))
        max_attn = self.linear_agg(x.reshape(B,C,D*H*W).max(2)[0])  # B C
        attn = F.sigmoid(max_attn+avg_attn)[:,:,None,None,None]  # B C,1,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_DCAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_DCAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D,H*W).mean(3).permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)
        max_attn = self.linear_agg(x.reshape(B,C,D,H*W).max(3)[0].permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)  # B C D
        attn = F.sigmoid(max_attn+avg_attn)[:,:,:,None,None]  # B C,D,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pixel_conv = nn.Conv2d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.reshape(B,C*D,H,W).max(1, keepdim=True)[0]
        avg_attn = x.reshape(B,C*D,H,W).mean(1, keepdim=True)  # B 1 H W
        attn = F.sigmoid(self.pixel_conv(torch.cat([max_attn, avg_attn], dim=1)))[:,:,None,:,:]  # B 1,1,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PDAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PDAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.spatial_conv = nn.Conv3d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.max(1, keepdim=True)[0]
        avg_attn = x.mean(1, keepdim=True)  # B 1 D H W
        attn = F.sigmoid(self.spatial_conv(torch.cat([max_attn, avg_attn], dim=1)))  # B 1,D,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class Deconv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=4, **kwargs):
        super(Conv2d, self).__init__()
        if gn == 'IN':
            bn = 'IN'
        else:
            bn = not gn
        # bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        # self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        if bn == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum)
        elif bn:
            # print("con2d bn")
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        else:
            # print("con2d gn")
            self.bn = None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu




class reg2d_large(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D', gn=False):
        super(reg2d_large, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,5,5), pad=(0,2,2), gn=gn)
        self.conv0_ = getattr(module, conv_name)(base_channel, base_channel, gn=gn)
        self.conv0_ = getattr(module, conv_name)(base_channel, base_channel, gn=gn)

        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,5,5), stride=(1,2,2), pad=(0,2,2), gn=gn)
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2, gn=gn)
        self.conv2_ = getattr(module, conv_name)(base_channel * 2, base_channel * 2, gn=gn)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4, gn=gn)
        self.conv4_ = getattr(module, conv_name)(base_channel * 4, base_channel * 4, gn=gn)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8, gn=gn)
        self.conv6_ = getattr(module, conv_name)(base_channel * 8, base_channel * 8, gn=gn)

        if gn == "IN":
            normlayer1 = nn.InstanceNorm3d(base_channel * 4)
            normlayer2 = nn.InstanceNorm3d(base_channel * 2)
            normlayer3 = nn.InstanceNorm3d(base_channel * 1)
        elif gn:
            normlayer1 = nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4)
            normlayer2 = nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2)
            normlayer3 = nn.GroupNorm(int(max(1, base_channel / 4)), base_channel)
        else:
            normlayer1 = nn.BatchNorm3d(base_channel * 4)
            normlayer2 = nn.BatchNorm3d(base_channel * 2)
            normlayer3 = nn.BatchNorm3d(base_channel * 1)


        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer1,
            # nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4) if gn else nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer2,
            # nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2) if gn else nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer3,
            # nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0_(self.conv0(x))
        conv2 = self.conv2_(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4_(self.conv4(self.conv3(conv2)))
        x = self.conv6_(self.conv6(self.conv5(conv4)))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)


class reg2d(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D', gn=False):
        super(reg2d, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,3,3), pad=(0,1,1), gn=gn)

        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2, gn=gn)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4, gn=gn)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8, gn=gn)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4) if gn else nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2) if gn else nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)

class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3, gn=False):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1, gn=gn)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels*2, kernel_size=3, stride=2, pad=1, gn=gn)
        self.conv2 = ConvBnReLU3D(base_channels*2, base_channels*2, gn=gn)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels*2, base_channels*4, kernel_size=3, stride=2, pad=1, gn=gn)
            self.conv4 = ConvBnReLU3D(base_channels*4, base_channels*4, gn=gn)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels*4, base_channels*8, kernel_size=3, stride=2, pad=1, gn=gn)
            self.conv6 = ConvBnReLU3D(base_channels*8, base_channels*8, gn=gn)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.GroupNorm(int(max(1, base_channels * 4 / 4)), base_channels * 4) if gn else nn.BatchNorm3d(base_channels * 4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.GroupNorm(int(max(1, base_channels * 2 / 4)), base_channels * 2) if gn else nn.BatchNorm3d(base_channels * 2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.GroupNorm(int(max(1, base_channels / 4)), base_channels) if gn else nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.down_size==3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        elif self.down_size==2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        return x.squeeze(1)  # B D H W






class line_cross_enhanced(nn.Module):
    def __init__(self, in_channels, base_channels, sample_num, cost_channel=1, gn=False):
        super(line_cross_enhanced, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, in_channels, pad=1, gn=gn)
        self.conv0_ = ConvBnReLU3D(sample_num*cost_channel, base_channels, pad=1, gn=gn)
        self.conv1_ = ConvBnReLU3D(base_channels, in_channels, pad=1, gn=gn)
        # self.prob = nn.Conv3d(in_channels, 1, 1, stride=1, padding=0)
    def forward(self, cost, sample_costs):
        # print(x)

        x = self.conv0(cost)
        x_ = self.conv1_(self.conv0_(sample_costs))
        x = x + x_
        # pro = self.prob(x)
        return x, x



class P_1to8_FeatureNet_4stage_Decoder_3DCNN(nn.Module):
    def __init__(self, base_channels=8, out_channel=[32,16,8], sample_num=[9,5,3,3], selfcross_weight=False, gn=False):
        super(P_1to8_FeatureNet_4stage_Decoder_3DCNN, self).__init__()
        print("P_1to8_FeatureNet_4stage_Decoder_3DCNN", gn, selfcross_weight,sample_num)
        self.sample_num = sample_num
        self.base_channels = base_channels
        self.selfcross_weight=selfcross_weight
        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8
        if selfcross_weight:
            self.out1_3dconv1 = ConvBnReLU3D(base_channels * 8, base_channels * 8, kernel_size=(3, 1, 1), pad=(1, 0, 0), gn=gn)
            self.out1_3dconv2 = ConvBnReLU3D(base_channels * 8, base_channels * 8, kernel_size=(3, 1, 1), pad=(1, 0, 0), gn=gn)
        else:
            if sample_num[0]>=5:
                self.out1_3dconv1 = ConvBnReLU3D(base_channels * 8, base_channels * 8, kernel_size=(3,1,1), pad=(1,0,0), gn=gn)
                self.out1_3dconv2 = ConvBnReLU3D(base_channels * 8, base_channels * 8, kernel_size=(sample_num[0],1,1), pad=0, gn=gn)
            else:
                self.out1_3dconv1 = ConvBnReLU3D(base_channels * 8, base_channels * 8, kernel_size=(1,1,1), pad=0, gn=gn)
                self.out1_3dconv2 = ConvBnReLU3D(base_channels * 8, base_channels * 8, kernel_size=(sample_num[0],1,1), pad=0, gn=gn)
        if selfcross_weight:
            self.out2_3dconv1 = ConvBnReLU3D(final_chs, final_chs, kernel_size=(3, 1, 1), pad=(1, 0, 0), gn=gn)
            self.out2_3dconv2 = ConvBnReLU3D(final_chs, final_chs, kernel_size=(3, 1, 1), pad=(1, 0, 0), gn=gn)
        else:
            if sample_num[1] >= 5:
                self.out2_3dconv1 = ConvBnReLU3D(final_chs, final_chs, kernel_size=(3,1,1), pad=(1,0,0), gn=gn)
                self.out2_3dconv2 = ConvBnReLU3D(final_chs, final_chs, kernel_size=(sample_num[1],1,1), pad=0, gn=gn)
            else:
                self.out2_3dconv1 = ConvBnReLU3D(final_chs, final_chs, kernel_size=(1,1,1), pad=0, gn=gn)
                self.out2_3dconv2 = ConvBnReLU3D(final_chs, final_chs, kernel_size=(sample_num[1],1,1), pad=0, gn=gn)



        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

    def forward(self, conv0, conv1, conv2, conv3, line_stage):
        intra_feat = conv3
        # print(intra_feat.shape)
        batch, channels, height, width = intra_feat.shape
        batch, sample_stage1, _, _ = line_stage["stage1"].shape
        intra_feat = intra_feat.to(torch.float32)
        intra_feat = F.grid_sample(intra_feat, line_stage["stage1"].view(batch, sample_stage1 * height, width, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        intra_feat = intra_feat.view(batch, channels, sample_stage1, height, width)
        # print(intra_feat.shape)
        intra_feat_3d = self.out1_3dconv1(intra_feat)
        intra_feat_3d = self.out1_3dconv2(intra_feat_3d)
        if self.selfcross_weight:
            intra_feat_3d = F.softmax(intra_feat_3d, dim=2)
            intra_feat = torch.sum(intra_feat * intra_feat_3d, 2)
            intra_feat = intra_feat.squeeze(2)
        else:
            intra_feat = intra_feat_3d.squeeze(2)
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv2)
        batch, channels, height, width = intra_feat.shape
        batch, sample_stage2, _, _ = line_stage["stage2"].shape
        # print(line_stage["stage2"].shape)
        intra_feat = intra_feat.to(torch.float32)
        intra_feat = F.grid_sample(intra_feat, line_stage["stage2"].view(batch, sample_stage2 * height, width, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        intra_feat = intra_feat.view(batch, channels, sample_stage2, height, width)
        # print(intra_feat.shape)
        intra_feat_3d = self.out2_3dconv1(intra_feat)
        # print(intra_feat_3d.shape)
        intra_feat_3d = self.out2_3dconv2(intra_feat_3d)
        # print(intra_feat.shape,intra_feat_3d.shape)

        if self.selfcross_weight:
            intra_feat_3d = F.softmax(intra_feat_3d, dim=2)
            # print(intra_feat.shape, intra_feat_3d.shape)
            intra_feat = torch.sum(intra_feat * intra_feat_3d, 2)

            intra_feat = intra_feat.squeeze(2)
        else:
            intra_feat = intra_feat_3d.squeeze(2)

        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv1)

        out = self.out3(intra_feat)
        outputs["stage3"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner3(conv0)
        out = self.out4(intra_feat)
        outputs["stage4"] = out

        return outputs




class FPNEncoder_selfcross(nn.Module):
    def __init__(self, base_channels, gn=False):
        super(FPNEncoder_selfcross, self).__init__()
        print("FPNEncoder_selfcross", base_channels)
        self.base_channels = base_channels
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        outputs = {}

        outputs["stage1"] = conv3

        outputs["stage2"] = conv2

        outputs["stage3"] = conv1

        outputs["stage4"] = conv0

        return outputs


class Visi_Net(nn.Module):
    def __init__(self, base_channels=4, gn=False, ref_channel=32, corr_channel=32, stage=1, use_ref=False, use_up=False):
        super(Visi_Net, self).__init__()
        self.base_channels = base_channels
        self.stage = stage
        self.use_up = use_up
        self.use_ref = use_ref
        final_ch = base_channels * (stage+1)
        self.conv0_channel = base_channels

        if self.use_ref:
            self.conv_ref = nn.Sequential(
                Conv2d(ref_channel, base_channels, 3, 1, padding=1, gn=gn),
                Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
            )
            self.conv0_channel = self.conv0_channel + base_channels
        if self.use_up:
            self.vis_up_layer = nn.Sequential(
                Conv2d(1, base_channels, 3, 1, padding=1, gn=gn),
                Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
            )
            self.conv0_channel = self.conv0_channel + base_channels

        self.conv_corr = nn.Sequential(
            Conv2d(corr_channel, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv0 = nn.Sequential(
            Conv2d(self.conv0_channel, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )
        self.inner0 = nn.Conv2d(self.conv0_channel, final_ch, 1, bias=True)
        self.out = nn.Conv2d(final_ch, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if stage>=2:
            self.conv1 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            )
            self.inner1 = nn.Conv2d(base_channels * 2, final_ch, 1, bias=True)

            if stage == 3:
                self.conv2 = nn.Sequential(
                    Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
                    Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
                    Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
                )
                self.inner2 = nn.Conv2d(base_channels * 4, final_ch, 1, bias=True)




    def forward(self, reference, corr, up_vis=None):
        x = self.conv_corr(corr)
        if self.use_ref:
            reference = self.conv_ref(reference)
            x = torch.cat([x, reference], dim=1)
        if self.use_up:
            up_vis = self.vis_up_layer(up_vis)
            x = torch.cat([x, up_vis], dim=1)

        # reference = self.conv_ref(reference)
        # corr = self.conv_corr(corr)
        # x = reference + corr
        # x = torch.cat([reference,corr],dim=1)
        if self.stage == 1:
            conv0 = self.conv0(x)
            intra = conv0
            intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner0(x)
            # cor_weight = self.out(intra)
        if self.stage == 2:
            conv0 = self.conv0(x)
            conv1 = self.conv1(conv0)
            intra = conv1
            intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv0)
            intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner0(x)
        if self.stage == 3:
            conv0 = self.conv0(x)
            conv1 = self.conv1(conv0)
            conv2 = self.conv2(conv1)
            intra = conv2
            intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv1)
            intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv0)
            intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner0(x)

        cor_weight = self.sigmoid(self.out(intra))

        return cor_weight.squeeze(1) #B H W


class stagenet_selfcross_regcorr(nn.Module):
    def __init__(self, inverse_depth=False, attn_fuse_d=True, attn_temp=2, use_visi_net=False, sample_num=1, using_reg=False, using_regcorr=True):
        super(stagenet_selfcross_regcorr, self).__init__()
        self.inverse_depth = inverse_depth
        self.use_visi_net = use_visi_net
        self.attn_fuse_d = attn_fuse_d
        self.attn_temp = attn_temp
        self.sample_num = sample_num
        self.using_reg = using_reg
        self.using_regcorr = using_regcorr
        print("self.using_regcorr",self.using_regcorr)
        self.tmp = [10.0, 10.0, 10.0, 1.0]

    def forward(self, ref_features, src_features, proj_matrices, depth_hypo, all_self_line_stages_grid, regnet, line_cross_enhanced_net ,stage_idx, vis_net=None,
                group_cor=False, group_cor_dim=8, split_itv=1, stageid=1):

        # step 1. feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_features[0].shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0
        # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        cor_weights = []
        self_line_stages_grid = [grid["stage{}".format(stage_idx + 1)] for grid in all_self_line_stages_grid]
        # step 2. Epipolar Transformer Aggregation
        for src_idx, (ref_feat, src_fea, src_proj, sample_stages_grids) in enumerate(zip(ref_features, src_features, src_projs, self_line_stages_grid)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W
            # ref_volume = ref_feat.to(torch.float32)
            # print(ref_volume.shape)
            ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)
            ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)

            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src) ** 2  # B C D H W
            del warped_src, src_proj, src_fea

            if self.using_regcorr:
                B, G, D, H, W = cor_feat.shape
                cor_feat_ = cor_feat.view(B, G * D, H, W)
                N_sample = sample_stages_grids.shape[1]
                # print(sample_stages_grids.shape)
                cor_feat_ = F.grid_sample(cor_feat_,
                                                   sample_stages_grids.reshape(B, N_sample * H, W, 2),
                                                   mode='bilinear',
                                                   padding_mode='zeros', align_corners=True)
                cor_feat_ = cor_feat_.view(B, G * D, N_sample, H, W)
                cor_feat_ = cor_feat_.reshape(B, G, D, N_sample, H, W)
                cor_feat_ = cor_feat_.permute(0, 1, 3, 2, 4, 5).reshape(B, G*N_sample, D, H, W)
                cor_feat, pro = line_cross_enhanced_net(cor_feat, cor_feat_)
                del cor_feat_, sample_stages_grids
            ###########


            if self.use_visi_net:
                # print("test")
                cor_feat_fuse = cor_feat.reshape(B, group_cor_dim * D, H, W)
                cor_weight = vis_net(ref_feature, cor_feat_fuse)  # B H W
                cor_weights.append(cor_weight)
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
                del cor_feat
            else:
                if not self.attn_fuse_d:
                    cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                    cor_weight_sum += cor_weight  # B H W
                    cor_weights.append(cor_weight)
                    cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
                else:
                    cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                    cor_weight_sum += cor_weight  # B D H W
                    cor_weight_save = cor_weight.max(1)[0]
                    cor_weights.append(cor_weight_save)
                    cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
                del cor_weight, cor_feat
        if self.use_visi_net:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            if not self.attn_fuse_d:
                cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
            else:
                cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W

        del cor_weight_sum, src_features

        # step 3. regularization
        attn_weight = regnet(cor_feats)  # B D H W
        attn_weight_ = attn_weight.clone()
        del cor_feats
        attn_weight = F.softmax(attn_weight, dim=1)  # B D H W

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W

        depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W
        # if self.training:
        #     depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W
        # else:
        #     depth = depth_regression(F.softmax(attn_weight_ * self.tmp[stageid], dim=1), depth_values=depth_hypo)

        if not self.training:
            with torch.no_grad():
                photometric_confidence = attn_weight.max(1)[0]  # B H W
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),
                                                       scale_factor=2 ** (3 - stage_idx), mode='bilinear',
                                                       align_corners=True).squeeze(1)
        else:
            photometric_confidence = torch.tensor(0.0, dtype=torch.float32, device=ref_features[0].device,
                                                  requires_grad=False)

        ret_dict = {"depth": depth, "photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo,
                    "attn_weight": attn_weight, "cor_weights": cor_weights, "cor_weights_": cor_weights}

        if self.inverse_depth:
            last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
            inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth

        return ret_dict




def upsample_depth(depth, mask, range=5):
    """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
    N, R, D, H, W = depth.shape
    mask = mask.view(N, range*range*D, H, W)
    mask = torch.softmax(mask, dim=1)
    depth = depth.view(N, range*range*D, H, W)
    # up_flow = F.unfold(depth, [3, 3], padding=1)
    # up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    depth = torch.sum(mask * depth, dim=1)
    return depth

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth




def self_p_line_normal_new(ref_fea,src_e, ref_e, ref_i, sample_num, sample_inter, mode="ref"):
    batch, channels = ref_fea.shape[0], ref_fea.shape[1]
    height, width = ref_fea.shape[2], ref_fea.shape[3]
    with torch.no_grad():
        proj = torch.matmul(ref_e, torch.inverse(src_e))
        # print(proj[0])
        xyz_src = torch.zeros([4], dtype=torch.float32).to(ref_fea.device)
        xyz_src[3] = 1.0
        xyz_src = torch.unsqueeze(xyz_src, 0).repeat(batch, 1)
        # print(xyz_src.shape,proj.shape)
        # print(xyz_src[0])
        xyz_src2ref = torch.matmul(proj,xyz_src.unsqueeze(2))[:,:3]
        # print(xyz_src2ref[0])
        xyz_src2ref[:,2:3] = torch.where(xyz_src2ref[:,2:3] == 0, xyz_src2ref[:,2:3] + 1e-10,
                                             xyz_src2ref[:,2:3])
        xyz_src2ref = xyz_src2ref/xyz_src2ref[:,2:3]

        pixel_src2ref = torch.matmul(ref_i, xyz_src2ref)[:,:2,0]
        # print(pixel_src2ref[0])
        pixel_src2ref_x = pixel_src2ref[:, 0:1].repeat(1, height * width)
        pixel_src2ref_y = pixel_src2ref[:, 1:2].repeat(1, height * width)
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=ref_fea.device)])
        y, x = y.contiguous(), x.contiguous()

        y, x = y.view(1, height * width).repeat(batch, 1), x.view(1, height * width).repeat(batch, 1)
        # print(y.shape)
        # print(y[0,32 * 80 + 40],x[0,32 * 80 + 40])

        y_ = y - pixel_src2ref_y
        x_ = x - pixel_src2ref_x


        # normal = torch.sqrt(torch.pow(y_, 2) + torch.pow(x_, 2))
        # y_normal = y_/normal
        # x_normal = x_/normal

        if x_.mean() != x_.mean():
            print("self_line_error")
            print(xyz_src2ref.mean())
        # print(y_[0,32 * 80 + 40], x_[0,32 * 80 + 40])

        normal = torch.sqrt(torch.pow(y_, 2) + torch.pow(x_, 2))

        line_x_normal = x_ / normal
        line_y_normal = y_ / normal

        line_x_normal = torch.where(line_x_normal>=0, line_x_normal, -1 * line_x_normal)
        line_y_normal = torch.where(line_x_normal>=0, line_y_normal, -1 * line_y_normal)

        new_interval_x = sample_inter * line_x_normal
        new_interval_y = sample_inter * line_y_normal




        x_min = (x - sample_num//2 * new_interval_x)
        y_min = (y - sample_num // 2 * new_interval_y)

        # print(x_min[0,:])
        x_range_samples = x_min.unsqueeze(1) + (torch.arange(0, sample_num, device=ref_fea.device,
                                                                         dtype=ref_fea.dtype,
                                                                         requires_grad=False).reshape(1, -1, 1) * new_interval_x.unsqueeze(1))
        y_range_samples = y_min.unsqueeze(1) + (torch.arange(0, sample_num, device=ref_fea.device,
                                                                         dtype=ref_fea.dtype,
                                                                         requires_grad=False).reshape(1, -1, 1) * new_interval_y.unsqueeze(1))

        # print(line_a[0,32 * 80 + 40], line_b[0,32 * 80 + 40])
        self_p_line_x = x_range_samples.view(batch,sample_num,height * width)

        self_p_line_y = y_range_samples.view(batch,sample_num,height * width)

        proj_xyz_grid = torch.stack((self_p_line_x, self_p_line_y, torch.ones_like(self_p_line_x)), dim=2)
        # print(pixel_src2ref_y[0,0], pixel_src2ref_x[0,0])
        # print(self_p_line_x[0, :, 32 * 80 + 40])
        # print(self_p_line_y[0,:,32*80 + 40])
        self_p_line_x = self_p_line_x / ((width - 1) / 2) - 1
        self_p_line_y = self_p_line_y / ((height - 1) / 2) - 1
        proj_xy = torch.stack((self_p_line_x, self_p_line_y), dim=3)  # [B, 5, H*W, 2]

    return proj_xyz_grid, proj_xy


class cost_up(nn.Module):
    def __init__(self, in_channels, base_channels, gn=False, using_pro=False):
        super(cost_up, self).__init__()

        self.conv0 = ConvBnReLU3D(in_channels, base_channels, pad=1, gn=gn)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels, stride=(1,2,2), pad=1, gn=gn)
        if using_pro:
            self.conv_cost = ConvBnReLU3D(1, base_channels, pad=1, gn=gn)
        else:
            self.conv_cost = ConvBnReLU3D(base_channels, base_channels, pad=1, gn=gn)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2, pad=1, gn=gn)
        # self.conv3 = Deconv3d(base_channels * 2, base_channels, stride=(1,2,2), padding=1, output_padding=(0,1,1), norm_type=norm_type)
        if gn == "IN":
            normlayer = nn.InstanceNorm3d(base_channels)
        elif gn:
            print("cost_up gn")
            normlayer = nn.GroupNorm(int(max(1, base_channels / 4)), base_channels)
        else:
            normlayer = nn.BatchNorm3d(base_channels)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer,
            # nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))
        # self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
    def forward(self, x, IGEV_cost):

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        IGEV_cost_ = self.conv_cost(IGEV_cost)
        conv2 = self.conv2(torch.cat([conv1, IGEV_cost_], dim=1))
        # print(conv2.shape)
        conv3 = self.conv3(conv2)
        # print(x.shape, conv1.shape, IGEV_cost.shape, conv3.shape)
        pro = conv3 + conv0
        # x_ = self.prob(pro)
        return pro, pro

class feature_up(nn.Module):
    def __init__(self, in_channels, in_channels_last, base_channels, gn='BN'):
        super(feature_up, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, pad=1, gn=gn)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels, pad=1, gn=gn)
        self.conv_fea = ConvBnReLU3D(in_channels_last, base_channels, pad=1, gn=gn)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2, pad=1, gn=gn)
        self.conv3 = ConvBnReLU3D(base_channels * 2, in_channels, pad=1, gn=gn)
        # self.conv3 = Deconv3d(base_channels * 2, base_channels, stride=(1,2,2), padding=1, output_padding=(0,1,1), norm_type=norm_type)
        # if gn == "IN":
        #     normlayer = nn.InstanceNorm3d(base_channels)
        # elif gn:
        #     normlayer = nn.GroupNorm(int(max(1, base_channels / 4)), base_channels)
        # else:
        #     normlayer = nn.BatchNorm3d(base_channels)
        # self.conv3 = nn.Sequential(
        #     nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
        #     normlayer,
        #     # nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
        #     nn.ReLU(inplace=True))
        # self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
    def forward(self, x, x_last):
        # print(x_last.shape)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv_fea = self.conv_fea(x_last)
        conv2 = self.conv2(torch.cat([conv1, conv_fea], dim=1))
        # print(conv2.shape)
        conv3 = self.conv3(conv2)
        # print(x.shape, conv1.shape, IGEV_cost.shape, conv3.shape)
        # pro = conv3 + conv0
        # x_ = self.prob(pro)
        return conv3, conv3


class Depth_up_module(nn.Module):
    def __init__(self, in_channels=8):
        super(Depth_up_module, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * 2 * 9, 1, padding=0))

    def forward(self, depth, cost_volume):
        b,c,d,h,w = cost_volume.shape
        cost_volume_ = cost_volume.view(b,c*d,h,w)
        mask = .25 * self.mask(cost_volume_)
        depth = upsample_depth(depth, mask)
        return depth


class Depth_up_module_new(nn.Module):
    def __init__(self, in_channels=8, base_channels=8,gn=False):
        super(Depth_up_module_new, self).__init__()
        print("Depth_up_module_new")
        self.depth_fea = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.Conv2d(base_channels, base_channels, 3, padding=1))
        self.ref_fea_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.mask = nn.Sequential(
            Conv2d(base_channels*2, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            # nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, 2 * 2 * 9, 1, padding=0))

    def forward(self, depth, ref_fea):
        b,_,h,w = ref_fea.shape
        depth_fea = self.depth_fea(depth.unsqueeze(1))
        depth_fea = F.interpolate(depth_fea, size=[h, w], mode='nearest')
        ref_fea = self.ref_fea_conv(ref_fea)
        fea_concat = torch.cat([depth_fea, ref_fea], dim=1)
        mask = .25 * self.mask(fea_concat)
        depth = upsample_depth(depth, mask)
        return depth

def upsample_depth(depth, mask, ratio=2):
    """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
    N, H, W = depth.shape

    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth.unsqueeze(1), [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)




class reg2d_large_cs(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D', gn=False):
        super(reg2d_large_cs, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,5,5), pad=(0,2,2), gn=gn)
        self.conv0_ = getattr(module, conv_name)(base_channel, base_channel, gn=gn)
        # self.conv0_ = getattr(module, conv_name)(base_channel, base_channel, gn=gn)

        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,5,5), stride=(1,2,2), pad=(0,2,2), gn=gn)
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2, gn=gn)
        self.conv2_ = getattr(module, conv_name)(base_channel * 2, base_channel * 2, gn=gn)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4, gn=gn)
        self.conv4_ = getattr(module, conv_name)(base_channel * 4, base_channel * 4, gn=gn)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8, gn=gn)
        self.conv6_ = getattr(module, conv_name)(base_channel * 8, base_channel * 8, gn=gn)



        if gn == "IN":
            normlayer1 = nn.InstanceNorm3d(base_channel * 4)
            normlayer2 = nn.InstanceNorm3d(base_channel * 2)
            normlayer3 = nn.InstanceNorm3d(base_channel * 1)
        elif gn:
            normlayer1 = nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4)
            normlayer2 = nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2)
            normlayer3 = nn.GroupNorm(int(max(1, base_channel / 4)), base_channel)
            print(gn)
            if gn:
                print(1)
            else:
                print(0)
        else:
            normlayer1 = nn.BatchNorm3d(base_channel * 4)
            normlayer2 = nn.BatchNorm3d(base_channel * 2)
            normlayer3 = nn.BatchNorm3d(base_channel * 1)


        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer1,
            # nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4) if gn else nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer2,
            # nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2) if gn else nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            normlayer3,
            # nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0_(self.conv0(x))
        conv2 = self.conv2_(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4_(self.conv4(self.conv3(conv2)))
        x = self.conv6_(self.conv6(self.conv5(conv4)))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        prob = self.prob(x)

        return prob.squeeze(1), x



class reg2d_cs(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D', gn=False):
        super(reg2d_cs, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,3,3), pad=(0,1,1), gn=gn)

        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2, gn=gn)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4, gn=gn)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1), gn=gn)
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8, gn=gn)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.GroupNorm(int(max(1, base_channel*4 / 4)), base_channel*4) if gn else nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.GroupNorm(int(max(1, base_channel*2 / 4)), base_channel*2) if gn else nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.GroupNorm(int(max(1, base_channel / 4)), base_channel) if gn else nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        prob = self.prob(x)

        return prob.squeeze(1), x





