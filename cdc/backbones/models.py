'''
@File  :model.py
@Date  :2023/1/29 20:01
@Desc  :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.cuda.amp import autocast
from cdc.backbones.resnet import resnet18, resnet34, resnet50
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
from cdc.losses.losses import CCLoss, DivClustLoss
import math


class ClusteringModel(nn.Module):
    def __init__(self, cfg):
        super(ClusteringModel, self).__init__()
        if cfg['backbone']['name'].startswith("resnet"):
            if cfg['backbone']['name'] == "resnet18":
                self.backbone = resnet18(cfg['method'])
            elif cfg['backbone']['name'] == "resnet34":
                self.backbone = resnet34(cfg['method'])
            elif cfg['backbone']['name'] == "resnet50":
                self.backbone = resnet50(cfg['method'])
            self.backbone.fc = nn.Identity()
            self.backbone_dim = self.backbone.inplanes
            cifar = cfg['data']['dataset'] in ["cifar10", "cifar20", "cifar100"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()

        self.nheads = cfg['backbone']['nheads']
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([
            nn.Sequential(
                          nn.Linear(self.backbone_dim, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True),
                        nn.Linear(512, cfg['backbone']['nclusters'])
        ) for _ in range(self.nheads)])

        self.projector_q = nn.Sequential(
            nn.Linear(self.backbone_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        self.projector_classify = nn.Sequential(
                        nn.Linear(256, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, cfg['backbone']['nclusters'])
        )
        
        self.classify_tail = nn.Sequential(
                          nn.Linear(self.backbone_dim, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True),
                        nn.Linear(512, cfg['backbone']['nclusters'])
        )
        self.classify_medium = nn.Sequential(
                          nn.Linear(self.backbone_dim, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True),
                        nn.Linear(512, cfg['backbone']['nclusters'])
        )

        
        

    def forward(self, x, forward_pass='default', dropout = None):
        if forward_pass == 'default':
            out = [cluster_head(self.backbone(x)) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            x = self.backbone(x)
            if dropout is not None:
                x = nn.Dropout(dropout)(x)
            out = {'features': x,
                       'output': [cluster_head(x) for cluster_head in self.cluster_head]}
            
        elif forward_pass == 'backbone_propos':
            out = self.backbone(x)
            out = self.projector_q(out)
            

        elif forward_pass == 'head_propos':
            out = [self.projector_classify(x)]

        elif forward_pass == 'propos_all':
            fea = self.backbone(x)
            fea = self.projector_q(fea)
            output = self.projector_classify(fea)
            out = {'features': fea,
                       'output': [output] }

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out
    
class CCModel(nn.Module):
    def __init__(self, cfg):
        super(CCModel, self).__init__()
        if cfg['backbone']['name'].startswith("resnet"):
            if cfg['backbone']['name'] == "resnet18":
                self.backbone = resnet18(cfg['method'])
            elif cfg['backbone']['name'] == "resnet34":
                self.backbone = resnet34(cfg['method'])
            elif cfg['backbone']['name'] == "resnet50":
                self.backbone = resnet50(cfg['method'])
            self.backbone.fc = nn.Identity()
            self.backbone_dim = self.backbone.inplanes
            cifar = cfg['data']['dataset'] in ["cifar10", "cifar20", "cifar100"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()

        self.instance_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_dim, cfg['backbone']['feat_dim']),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_dim, cfg['backbone']['nclusters']),
            nn.Softmax(dim=1)
        )
        
        

    def forward(self, x_i, x_j=None, forward_pass='default'):
        if forward_pass == 'default':
            x = torch.cat([x_i, x_j], dim=0)
            h = self.backbone(x)
            h_i, h_j = torch.chunk(h, 2, dim=0)
            z_i = F.normalize(self.instance_projector(h_i), dim=1)
            z_j = F.normalize(self.instance_projector(h_j), dim=1)

            c_i = self.cluster_projector(h_i)
            c_j = self.cluster_projector(h_j)
            return z_i, z_j, c_i, c_j
        
        elif forward_pass == 'test':
            h = self.backbone(x_i)
            c_i = self.cluster_projector(h)
            return [c_i]
        
        elif forward_pass == 'return_all':
            h = self.backbone(x_i)
            c_i = self.cluster_projector(h)
            logits = self.cluster_projector[:3](h) 
            return {'features': h,
                    'logits':[logits],
                    'output': [c_i]}
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

class DivClust_CCModel(nn.Module):
    def __init__(self, cfg):
        super(DivClust_CCModel, self).__init__()
        if cfg['backbone']['name'].startswith("resnet"):
            if cfg['backbone']['name'] == "resnet18":
                self.backbone = resnet18(cfg['method'])
            elif cfg['backbone']['name'] == "resnet34":
                self.backbone = resnet34(cfg['method'])
            elif cfg['backbone']['name'] == "resnet50":
                self.backbone = resnet50(cfg['method'])
            self.backbone.fc = nn.Identity()
            self.backbone_dim = self.backbone.inplanes
            cifar = cfg['data']['dataset'] in ["cifar10", "cifar20", "cifar100"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()

        self.NMI_target= cfg['NMI_target']
        self.NMI_interval = cfg['NMI_interval']
        self.threshold_rate=cfg['threshold_rate']
        self.divclust_mbank_size = cfg['divclust_mbank_size']
        self.clusterings = cfg['clusterings']
        self.clusters = cfg['backbone']['nclusters']

        self.instance_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_dim, cfg['backbone']['feat_dim']),
        )
        
        self.cluster_projector = nn.Sequential(
            MultiheadLinear(self.backbone_dim, self.backbone_dim, self.clusterings, True),
            nn.ReLU(),
            MultiheadLinear(self.backbone_dim, self.clusters, self.clusterings, True),
        )
        
        self.CCLoss = CCLoss()
        self.DivLoss = DivClustLoss(threshold=1, NMI_target= self.NMI_target, NMI_interval = self.NMI_interval, threshold_rate=self.threshold_rate, divclust_mbank_size = self.divclust_mbank_size)

        self.current_step = 0
        
    def forward(self, x1, x2=None, forward_pass='default'):
        if forward_pass == 'loss':
            f1, f2 = self.backbone(x1), self.backbone(x2)
            p1, p2 = F.softmax(self.cluster_projector(f1), dim=-1), F.softmax(self.cluster_projector(f2), dim=-1)
            z1, z2 = self.instance_projector(f1), self.instance_projector(f2)

            loss_ce, loss_ne, loss_cc = self.CCLoss(p1, p2, z1, z2)
            diversity_loss, threshold, _ = self.DivLoss(torch.cat([p1, p2], dim=1), self.current_step)
            loss_ce_sum = sum(loss_ce) /self.clusterings
            loss_ne_sum = sum(loss_ne) / self.clusterings
            diversity_loss = diversity_loss / self.clusterings
            loss = loss_ce_sum + loss_ne_sum + loss_cc + diversity_loss
            self.current_step+=1
            return loss, {"loss_cc": loss_cc, "loss_ce": loss_ce_sum, "loss_ne": loss_ne_sum, "loss_div": diversity_loss, "threshold": threshold}
        
        elif forward_pass == 'default':
            f1, f2 = self.backbone(x1), self.backbone(x2)
            p1, p2 = F.softmax(self.cluster_projector(f1), dim=-1), F.softmax(self.cluster_projector(f2), dim=-1)
            z1, z2 = self.instance_projector(f1), self.instance_projector(f2)
            return z1, z2, p1, p2
        
        elif forward_pass == 'return_all':
            f1 = self.backbone(x1)
            p1 = F.softmax(self.cluster_projector(f1), dim=-1)
            return {'features': f1,
                   'output': [p1]}
        

    @torch.no_grad()
    def predict(self, x, softmax=True, return_features=False):
        f = self.backbone(x)
        p = self.cluster_projector(f)
        if softmax:
            p = F.softmax(p, dim=-1)
        if return_features:
            return p, f
        else:
            return p

class MultiheadLinear(nn.Module):
    def __init__(self, d_in, d_out, parallel_no, bias=False, same_weight_init=False):
        super(MultiheadLinear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.parallel_no = parallel_no
        self.add_bias = bias

        weights = [torch.empty((d_out, d_in)) for k_ in range(parallel_no)]
        if self.add_bias:
            bias = [torch.empty((1, d_out)) for k_ in range(parallel_no)]
        else:
            bias = [None]*parallel_no
        initialized_weights, initialized_bias = [],[]
        for w, b in zip(weights, bias):
            w_init, b_init = self.init_weights(w, b)
            initialized_weights.append(w_init)
            initialized_bias.append(b_init)

        self.weight = nn.Parameter(torch.stack(initialized_weights))
        if same_weight_init:
            for k in range(parallel_no):
                self.weight.data[k] = self.weight.data[0]

        if self.add_bias:
            self.bias = nn.Parameter(torch.stack(initialized_bias))
            if same_weight_init:
                for k in range(parallel_no):
                    self.bias.data[k] = self.bias.data[0]
        else:
            self.bias = None

    def init_weights(self, weight, bias=None):
        from torch.nn.modules.linear import init
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(bias, -bound, bound)
        return weight, bias

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 2:
            r = torch.einsum('ik,bjk->bij', x, self.weight)
        elif len(x_shape) == 3:
            if x_shape[0] == 1:
                x = x.squeeze(0)
                r = torch.einsum('ik,bjk->bij', x, self.weight)
            else:
                r = torch.einsum('bik,bjk->bij', x, self.weight)
        if self.add_bias:
            return r + self.bias
        else:
            return r

class CaliMLP(nn.Module):
    def __init__(self, cfg):
        super(CaliMLP, self).__init__()
        if cfg['backbone']['name'] == "resnet50":
            self.backbone_dim = 2048
        else:
            self.backbone_dim = 512
        self.calibration_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg['backbone']['nclusters'])
        )

        self.calibration_propos = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cfg['backbone']['nclusters'])
        )
        
        self.calibration_tail = nn.Sequential(
                          nn.Linear(self.backbone_dim, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True),
                        nn.Linear(512, cfg['backbone']['nclusters'])
        )
        self.calibration_medium = nn.Sequential(
                          nn.Linear(self.backbone_dim, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True),
                        nn.Linear(512, cfg['backbone']['nclusters'])
        )
    def forward(self, x, forward_pass=None):
        if forward_pass == 'calibration':
            out = self.calibration_head(x)
        elif forward_pass == 'calibration_propos':
            out = self.calibration_propos(x)
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))
        return out
    
class TCLModel(nn.Module):
    def __init__(self, cfg, cmnist=False):
        super(TCLModel, self).__init__()
        if cfg['backbone']['name'].startswith("resnet"):
            if cfg['backbone']['name'] == "resnet18":
                if cmnist:
                    import torchvision.models as models
                    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
                    self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = resnet18(cfg['method'])
                self.hidden_dim = 512
            elif cfg['backbone']['name'] == "resnet34":
                self.backbone = resnet34(cfg['method'])
                self.hidden_dim = 512
            elif cfg['backbone']['name'] == "resnet50":
                self.backbone = resnet50(cfg['method'])
                self.hidden_dim = 2048
            self.backbone.fc = nn.Identity()
            self.backbone_dim = self.backbone.inplanes
            cifar = cfg['data']['dataset'] in ["cifar10", "cifar20", "cifar100"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        self.nheads = cfg['backbone']['nheads']       
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.feature_dim = 128
        self.cluster_num = cfg['backbone']['nclusters']
        hidden_dim = self.hidden_dim
        self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.cluster_num),
        ) 
        
        """ self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.cluster_num),
        )"""
        trunc_normal_(self.cluster_projector[2].weight, std=0.02)
        trunc_normal_(self.cluster_projector[5].weight, std=0.02) 

    def forward(self, x_i, x_j, return_ci=True):
        
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_j = self.cluster_projector(h_j)

        if return_ci:
            c_i = self.cluster_projector(h_i)
            return z_i, z_j, c_i, c_j
        else:
            return z_i, z_j, c_j

    def forward_c(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return c

    def forward_zc(self, x):
        h = self.backbone(x)
        z = F.normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return z, c
    
    def forward_osr(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        return c
    
    def forward_all(self, x):
        h = self.backbone(x)
        z = F.normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        cc = F.softmax(c, dim=1)
        return c, cc
    
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channel=3,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rep_dim = 512 * block.expansion
        self.fc = nn.Linear(self.rep_dim, self.rep_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # for name, param in self.named_parameters():
        #     if (
        #         name.startswith("conv1")
        #         or name.startswith("bn1")
        #         or name.startswith("layer1")
        #         or name.startswith("layer2")
        #     ):
        #         print("Freeze gradient for", name)
        #         param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # with torch.no_grad():
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def get_resnet(cfg):
    if cfg['backbone']['name'] == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2]), 512
    elif cfg['backbone']['name'] == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3]), 512
    elif cfg['backbone']['name'] == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3]), 2048
    else:
        raise NotImplementedError
    
class Network(nn.Module):
    def __init__(self, resnet, hidden_dim, feature_dim, class_num):
        super(Network, self).__init__()
        self.backbone = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.cluster_num),
        )
        trunc_normal_(self.cluster_projector[2].weight, std=0.02)
        trunc_normal_(self.cluster_projector[5].weight, std=0.02)

    def forward(self, x_i, x_j, return_ci=True):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_j = self.cluster_projector(h_j)

        if return_ci:
            c_i = self.cluster_projector(h_i)
            return z_i, z_j, c_i, c_j
        else:
            return z_i, z_j, c_j

    def forward_c(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return c

    def forward_zc(self, x):
        h = self.backbone(x)
        z = F.normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return z, c
        
class SCANModel(nn.Module):
    def __init__(self, cfg):
        super(SCANModel, self).__init__()
        if cfg['backbone']['name'].startswith("resnet"):
            if cfg['backbone']['name'] == "resnet18":
                self.backbone = resnet18(cfg['method'])
            elif cfg['backbone']['name'] == "resnet34":
                self.backbone = resnet34(cfg['method'])
            elif cfg['backbone']['name'] == "resnet50":
                self.backbone = resnet50(cfg['method'])
            self.backbone.fc = nn.Identity()
            self.backbone_dim = self.backbone.inplanes
            cifar = cfg['data']['dataset'] in ["cifar10", "cifar20", "cifar100"]
            usps = cfg['data']['dataset'] in ["usps"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        self.nheads = cfg['backbone']['nheads']
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, cfg['backbone']['nclusters']) for _ in range(self.nheads)])
        
         # Extra heads for long-tail experts
        self.classify_tail = nn.Linear(self.backbone_dim, cfg['backbone']['nclusters'])
        self.classify_medium = nn.Linear(self.backbone_dim, cfg['backbone']['nclusters'])

    def forward(self, x, forward_pass='default', dropout = None):
        if forward_pass == 'default':
            out = [cluster_head(self.backbone(x)) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            x = self.backbone(x)
            if dropout is not None:
                x = nn.Dropout(dropout)(x)
            out = {'features': x,
                       'output': [cluster_head(x) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out
    
    

    
    
    