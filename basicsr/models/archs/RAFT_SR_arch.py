import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from efficientVit import EfficientViT
from gru import BasicUpdateBlock_x4
from context_cnn import CNNEncoderLight
from transformer import FeatureTransformer
from raftutils import feature_add_position
#from cross_attention import CVIM 

class RAFT(nn.Module):
    def __init__(self, vit_config, gru_iters=8, feat_dim=64):
        super().__init__()
        self.gru_iters = gru_iters
        self.feat_dim = feat_dim
        self.backbone = EfficientViT(**vit_config)

        # this is a hard code here
        self.transformer = FeatureTransformer(num_layers=4, d_model=feat_dim, nhead=1, ffn_dim_expansion=2)
        self.gru = BasicUpdateBlock_x4(sterep_dim=feat_dim)
        self.context = CNNEncoderLight(output_dim=48*2)

    def upsample_flow_raft(self, stereo_img, mask):
        """ Upsample stereo images [H, W, 3] -> [H*4, W*4, 3] """
        N, _, H, W = stereo_img.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_stero = F.unfold(4 * stereo_img, [3,3], padding=1)
        up_stero = up_stero.view(N, -1, 9, 1, 1, H, W)

        up_stero = torch.sum(mask * up_stero, dim=2)
        up_stero = up_stero.permute(0, 1, 4, 2, 5, 3)
        return up_stero.reshape(N, -1, 4*H, 4*W)

    def forward(self, stereo_imgs,h_pad = None,w_pad =None):
       # N, _, H, W = left_img.shape
        # assert stereo_imgs.shape(1) == 6 # "输入的通道数必须是6，表示两张RGB图像"
        print("stereo_imgs.device",stereo_imgs.device)
        upsampled_stereos_list = []
        N, _, H, W = stereo_imgs.shape
        # 分割成左右视图
        left_img = stereo_imgs[:, :3, :, :]  # 取前三个通道为左视图
        right_img = stereo_imgs[:, 3:, :, :]  # 取后三个通道为右视图

        stereo_imgs = torch.cat((left_img, right_img), dim=0)
        stereo_imgs_upsampled = F.interpolate(stereo_imgs, scale_factor=8, mode='bilinear', align_corners=True)
        print("stereo_imgs_upsampled.device",stereo_imgs_upsampled.device)
        stereo_features = self.backbone(stereo_imgs_upsampled)  
        stereo_context = self.context(stereo_imgs_upsampled)

        # transformer enhancement with self-attention and cross-attention
        left_features, right_features = torch.split(stereo_features, [N, N], dim=0)
        
        left_features, right_features = feature_add_position(left_features, right_features, attn_splits=8, feature_channels=self.feat_dim)
        left_features, right_features = self.transformer(left_features, right_features, attn_num_splits=8)
        stereo_features = torch.cat((left_features, right_features), dim=0)
        #enhanced_left_features, enhanced_right_features = self.cvim(left_features, right_features)
        #enhanced_stereo_features = torch.cat((enhanced_left_features, enhanced_right_features), dim=0)

        # extract contextual features
        inp_features, hidden_features = torch.split(stereo_context, [48, 48], dim=1)
        inp_features = torch.relu(inp_features)
        hidden_features = torch.tanh(hidden_features)
        for i in range(self.gru_iters):
            hidden_features, mask = self.gru(stereo_features, hidden_features, inp_features)
            upsampled_stereo = self.upsample_flow_raft(stereo_imgs, mask)
            upsampled_stereo = torch.cat((upsampled_stereo[0], upsampled_stereo[1]), dim=0).unsqueeze(0)
            scale_factor = 4  # 根据您的模型放大因子进行调整
            if h_pad is not None:
                hp, wp = h_pad * scale_factor, w_pad* scale_factor
                upsampled_stereo = upsampled_stereo[:, :, :-(hp) if hp != 0 else None, :-(wp) if wp != 0 else None]

            upsampled_stereos_list.append(upsampled_stereo)

        return upsampled_stereos_list
            

def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")

import cv2
if __name__ == '__main__':
    # attach_debugger()
    EfficientViT_m0 = {
        'embed_dim': [32, 56],
        'depth': [2, 4],    # !!ori[1, 2]
        'num_heads': [4, 4],
        'window_size': [7, 7],
        'kernels': [5, 5, 5, 5],
        'down_ops' : [['subsample', 1], ['']]
    }
    left_img = torch.randn(1, 3, 128, 128).cuda()
    right_img = torch.randn(1, 3, 128, 128).cuda()
    inp = torch.randn(1,6,30,90).cuda()

    raft_sr_model = RAFT(EfficientViT_m0, feat_dim=56)
    import pdb; pdb.set_trace()
    output = raft_sr_model(inp)
    print(output.shape)
    from thop import profile
    # _, params = profile(raft_sr_model, inputs=(left_img, right_img))
    # print(params)
