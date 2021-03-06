import torch
import torch.nn as nn
import torch.nn.functional as F
from mesh2tex.layers import EqualizedLR
from mesh2tex.layers import ResnetBlockFC, ResnetBlockConv1D
from mesh2tex.geometry.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction


class SimplePointnet(nn.Module):
    def __init__(self, c_dim=128, hidden_dim=128,
                 leaky=False, eq_lr=False):
        super().__init__()
        # Attributes
        self.c_dim = c_dim
        self.eq_lr = eq_lr

        # Activation function
        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = avgpool

        # Submodules
        self.conv_p = nn.Conv1d(6, 2*hidden_dim, 1)
        self.conv_0 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.conv_1 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.conv_3 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)
            self.conv_2 = EqualizedLR(self.conv_2)
            self.conv_3 = EqualizedLR(self.conv_3)
            self.fc_c = EqualizedLR(self.fc_c)

    def forward(self, geometry):
        p = geometry['points']
        n = geometry['normals']

        # Encode position into batch_size x F x T
        pn = torch.cat([p, n], dim=1)
        net = self.conv_p(pn)

        # Always pool to batch_size x F x 1,
        # expand to batch_size x F x T
        # and concatenate to batch_size x 2F x T
        net = self.conv_0(self.actvn(net))
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv_1(self.actvn(net))
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv_2(self.actvn(net))
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled = pooled.expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv_3(self.actvn(net))

        # Recude to  batch_size x F
        net = self.pool(net, dim=2)

        c = self.fc_c(self.actvn(net))

        geom_descr = {
            'global': c,
        }

        return geom_descr


class ResnetPointnet(nn.Module):
    def __init__(self, c_dim=128, dim=6, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, geometry):
        p = geometry['points']
        n = geometry['normals']
        batch_size, T, D = p.size()

        pn = torch.cat([p, n], dim=1)
        # output size: B x T X F
        net = self.fc_pos(pn)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnetConv(nn.Module):
    def __init__(self, c_dim=128, dim=6, hidden_dim=128, pretrain = False):
        super().__init__()
        self.c_dim = c_dim
        self.pretrain = pretrain
        self.fc_pos = nn.Conv1d(dim, 2*hidden_dim, 1)
        self.block_0 = ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockConv1D(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, geometry):
        p = geometry['points']
        n = geometry['normals']
        batch_size, T, D = p.size()
        if self.pretrain == False:
            pn = torch.cat([p, n], dim=1)
        else:
            pn = p
        # output size: B x T X F
        net = self.fc_pos(pn)
        net = self.block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)
        
        # Recude to  B x F
        net = self.pool(net, dim=2)

        c = self.fc_c(self.actvn(net))

        geom_descr = {
            'global': c,
        }

        return geom_descr
class Pointnet2(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super(Pointnet2, self).__init__()
        in_channel = 3 
        self.normal_channel = True
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)

        '''
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 512)
        '''
    def forward(self, geometry):
        xyz = geometry['points']
        norm = geometry['normals']
        B, T, D = xyz.size()
                                                                                                                                            
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.fc1(x))
        
        #x = self.drop1(F.relu(self.fc1(x)))
        #x = self.drop2(F.relu(self.fc2(x)))
        #x = self.fc3(x)
        geom_descr = {
            'global': x,
        }
        return geom_descr

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def avgpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out
