import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64+64) -> None:
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))     # h hat
        h = (1 - z) * h + z * q

        # vertical 
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat((r*h, x), dim=1)))
        h = (1 - z) * h + z * q
        return h
    

class StereoEncoderLight(nn.Module):
    # redundant
    def __init__(self, input_dim=128, output_dim=64): # cor_planes = (2 * radius + 1) ** 2 * 4 =25*4
        super().__init__()
        # this is a hard code here
        self.convc1 = nn.Conv2d(input_dim, input_dim//2, 1, padding=0)
        self.convc2 = nn.Conv2d(input_dim//2, output_dim, 3, padding=1)
        
    def forward(self, stereo_feat):
        stereo_feat = F.relu(self.convc1(stereo_feat))
        stereo_feat = F.relu(self.convc2(stereo_feat))
        return stereo_feat
        

class BasicUpdateBlock_x4(nn.Module):
    def __init__(self, sterep_dim=128, hidden_dim=48, context_dim=48):
        super(BasicUpdateBlock_x4, self).__init__()
        self.encoder = StereoEncoderLight(sterep_dim, output_dim=hidden_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim+context_dim)

        self.mask = nn.Sequential(
            nn.Conv2d(context_dim, 116, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(116, 16*9, 1, padding=0))

    def forward(self, stereo_feat, net, inp, upsample=True):        
        stereo_feat = self.encoder(stereo_feat)
        inp = torch.cat((stereo_feat, inp), dim=1)
        net = self.gru(net, inp)    
        mask = .25 * self.mask(net)
        return net, mask