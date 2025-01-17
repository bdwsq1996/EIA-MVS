import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.module import *


class EIA(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8, 
                reg_channel=8, stage_splits=[8,8,4,4], depth_interals_ratio=[0.5,0.5,0.5,0.5],
                group_cor=False, group_cor_dim=[8,8,8,8],
                inverse_depth=False,
                agg_type='ConvBnReLU3D',
                attn_temp=2,
                attn_fuse_d=True,
                use_visi_net=False
                ):
        super(EIA, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.use_visi_net = use_visi_net

        self.using_gn = False
        self.self_line_corr = False
        self.stage_sample = [7,5,3,1]
        self.sample_inter = [1,1.5,2,2.5]
        self.selfcross_weight = False
        print("self.using_gn", self.using_gn)
        print("self_line_corr", self.self_line_corr)
        print("stage_sample", self.stage_sample)
        print("sample_inter", self.sample_inter)
        print("selfcross_weight", self.selfcross_weight)
        print("EIA-MVS")


        print("use_visi_net", use_visi_net)
        print("attn_fuse_d", attn_fuse_d)


        self.encoder = FPNEncoder_selfcross(base_channels=fpn_base_channel, gn=self.using_gn)
        feat_chs_list = [64,32,16,8]
        print("P_1to8_FeatureNet_4stage_Decoder_3DCNN", feat_chs_list)
        # 考虑使用计算weight
        self.decoder = P_1to8_FeatureNet_4stage_Decoder_3DCNN(base_channels=feat_chs_list[-1],
                                                              out_channel=feat_chs_list,
                                                              sample_num=self.stage_sample,
                                                              selfcross_weight=self.selfcross_weight, gn=self.using_gn)

        self.stagenet = stagenet_selfcross_regcorr(inverse_depth, attn_fuse_d, attn_temp, use_visi_net=use_visi_net, sample_num=self.stage_sample)

        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        if use_visi_net:
            self.vis_net = nn.ModuleList()
        else:
            self.vis_net = [None, None,None, None]

        self.line_cross_enhanced_net = nn.ModuleList()

        if reg_net == 'reg3d':
            self.down_size = [3,3,2,2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]

            self.line_cross_enhanced_net.append(line_cross_enhanced(group_cor_dim[idx], reg_channel, self.stage_sample[idx], cost_channel=in_dim, gn=self.using_gn))

            if reg_net == 'reg2d':
                self.reg.append(reg2d_large(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type, gn=self.using_gn))
                # self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type, gn=self.using_gn))
            elif reg_net == 'reg3d':
                self.reg.append(reg3d(in_channels=1, base_channels=reg_channel, down_size=self.down_size[idx]))
        
    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()

        features_encoder = []
        reference_features = []
        src_features = []
        V = len(imgs)
        for vi in range(V):
            img_v = imgs[vi]
            features_encoder.append(self.encoder(img_v))


        all_self_line_stages_grid = []
        for src_view_idx in range(V-1):
            self_line_stages = {}
            self_line_stages_grid = {}
            src_line_stages = {}
            for stage in range(self.num_stage):
                proj_matrices_curr_stage = proj_matrices["stage{}".format(stage + 1)]
                proj_matrices_curr_stage = torch.unbind(proj_matrices_curr_stage, 1)
                ref_proj_stage, src_projs_stage = proj_matrices_curr_stage[0], proj_matrices_curr_stage[1:]
                ref_proj = ref_proj_stage
                src_proj = src_projs_stage[src_view_idx]
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                self_line_grid, self_line = self_p_line_normal_new(features_encoder[0]["stage{}".format(stage + 1)],
                                               src_proj[:, 0, :4, :4], ref_proj[:, 0, :4, :4],
                                               ref_proj[:, 1, :3, :3]
                                               , self.stage_sample[stage], self.sample_inter[stage])

                self_line_stages["stage{}".format(stage + 1)] = self_line
                self_line_stages_grid["stage{}".format(stage + 1)] = self_line

                src_line_grid, src_line = self_p_line_normal_new(features_encoder[src_view_idx + 1]["stage{}".format(stage + 1)],
                                              ref_proj[:, 0, :4, :4], src_proj[:, 0, :4, :4], src_proj[:, 1, :3, :3]
                                              , self.stage_sample[stage], self.sample_inter[stage])

                src_line_stages["stage{}".format(stage + 1)] = src_line
            all_self_line_stages_grid.append(self_line_stages_grid)

            reference_features.append(self.decoder(features_encoder[0]["stage4"],
                                                           features_encoder[0]["stage3"],
                                                           features_encoder[0]["stage2"],
                                                           features_encoder[0]["stage1"],
                                                           self_line_stages
                                                           ))
            src_features.append(self.decoder(features_encoder[src_view_idx+1]["stage4"],
                                                           features_encoder[src_view_idx+1]["stage3"],
                                                           features_encoder[src_view_idx+1]["stage2"],
                                                           features_encoder[src_view_idx+1]["stage1"],
                                                           src_line_stages
                                                           ))
        # step 2. iter (multi-scale)
        outputs = {}
        for stage_idx in range(self.num_stage):

            features_stage_ref = [feat["stage{}".format(stage_idx + 1)] for feat in reference_features]
            features_stage_src = [feat["stage{}".format(stage_idx + 1)] for feat in src_features]
            # print(features_stage_src[0].shape)
            B, C, H, W = features_stage_src[0].shape

            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]

            
            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device, imgs[0][0].dtype, H, W)
                else:
                    depth_hypo = init_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device, imgs[0][0].dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach(), self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_interval = (depth_max - depth_min) / 192 
                    depth_hypo = schedule_range(outputs_stage['depth'].detach(), self.stage_splits[stage_idx], self.depth_interals_ratio[stage_idx] * depth_interval, H, W)
            
            outputs_stage = self.stagenet(features_stage_ref, features_stage_src, proj_matrices_stage, depth_hypo=depth_hypo,
                                    all_self_line_stages_grid=all_self_line_stages_grid, regnet=self.reg[stage_idx], line_cross_enhanced_net=self.line_cross_enhanced_net[stage_idx], stage_idx=stage_idx,
                                    group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                    split_itv=self.depth_interals_ratio[stage_idx], vis_net=self.vis_net[stage_idx], stageid=stage_idx)
            
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


def cross_entropy_loss(mask_true, hypo_depth, depth_gt, attn_weight):
    B, D, H, W = attn_weight.shape
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6
    gt_index_image = torch.argmin(torch.abs(hypo_depth - depth_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    gt_index_volume = torch.zeros_like(attn_weight).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(attn_weight + 1e-8), dim=1).squeeze(1)  # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    return masked_cross_entropy



def MVS4net_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    inverse = kwargs.get("inverse_depth", False)
    depth_fuse = kwargs.get("depth_fuse", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    depth_fuse_loss = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)


        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss



        if depth_fuse and stage_key >= 1:
            fuse_loss = F.smooth_l1_loss(depth_pred[mask], depth_gt[mask], reduction='mean')
            depth_fuse_loss.append(fuse_loss)
            total_loss = total_loss + fuse_loss

    return total_loss, stage_ce_loss, range_err_ratio, depth_fuse_loss




def Blend_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    inverse = kwargs.get("inverse_depth", False)
    depth_fuse = kwargs.get("depth_fuse", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    depth_fuse_loss = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)
        #
        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss




    #
    depth_interval = hypo_depth[:, 0, :, :] - hypo_depth[:, 1, :, :]

    abs_err = torch.abs(depth_gt[mask] - depth_pred[mask])
    abs_err_scaled = abs_err / (depth_interval[mask] * 192. / 128.)
    epe = abs_err_scaled.mean()
    err3 = (abs_err_scaled <= 3).float().mean()
    err1 = (abs_err_scaled <= 1).float().mean()
    return total_loss, stage_ce_loss, range_err_ratio, epe, err3, err1



