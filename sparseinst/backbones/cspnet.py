from functools import partial

import torch
import torch.nn as nn

from timm.models.layers import create_conv2d, create_act_layer
from timm.models.layers import DropPath, AvgPool2dSame, create_attn


from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import Backbone, BACKBONE_REGISTRY


model_cfgs = dict(
    cspresnet50=dict(
        stem=dict(out_chs=64, kernel_size=7, stride=2, pool='max'),
        stage=dict(
            out_chs=(128, 256, 512, 1024),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(2.,) * 4,
            bottle_ratio=(0.5,) * 4,
            block_ratio=(1.,) * 4,
            cross_linear=True,
        )
    ),
    cspresnet50d=dict(
        stem=dict(out_chs=[32, 32, 64], kernel_size=3, stride=2, pool='max'),
        stage=dict(
            out_chs=(128, 256, 512, 1024),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(2.,) * 4,
            bottle_ratio=(0.5,) * 4,
            block_ratio=(1.,) * 4,
            cross_linear=True,
        )
    ),
    cspresnet50w=dict(
        stem=dict(out_chs=[32, 32, 64], kernel_size=3, stride=2, pool='max'),
        stage=dict(
            out_chs=(256, 512, 1024, 2048),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            exp_ratio=(1.,) * 4,
            bottle_ratio=(0.25,) * 4,
            block_ratio=(0.5,) * 4,
            cross_linear=True,
        )
    ),
    cspresnext50=dict(
        stem=dict(out_chs=64, kernel_size=7, stride=2, pool='max'),
        stage=dict(
            out_chs=(256, 512, 1024, 2048),
            depth=(3, 3, 5, 2),
            stride=(1,) + (2,) * 3,
            groups=(32,) * 4,
            exp_ratio=(1.,) * 4,
            bottle_ratio=(1.,) * 4,
            block_ratio=(0.5,) * 4,
            cross_linear=True,
        )
    ),
    cspdarknet53=dict(
        stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
        stage=dict(
            out_chs=(64, 128, 256, 512, 1024),
            depth=(1, 2, 8, 8, 4),
            stride=(2,) * 5,
            exp_ratio=(2.,) + (1.,) * 4,
            bottle_ratio=(0.5,) + (1.0,) * 4,
            block_ratio=(1.,) + (0.5,) * 4,
            down_growth=True,
        )
    ),
    darknet53=dict(
        stem=dict(out_chs=32, kernel_size=3, stride=1, pool=''),
        stage=dict(
            out_chs=(64, 128, 256, 512, 1024),
            depth=(1, 2, 8, 8, 4),
            stride=(2,) * 5,
            bottle_ratio=(0.5,) * 5,
            block_ratio=(1.,) * 5,
        )
    )
)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, aa_layer=None,
                 drop_block=None):
        super(ConvBnAct, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        self.bn = norm_layer(out_channels)
        self.act = act_layer()
        self.aa = aa_layer(
            channels=out_channels) if stride == 2 and use_aa else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


def create_stem(
        in_chans=3, out_chs=32, kernel_size=3, stride=2, pool='',
        act_layer=None, norm_layer=None, aa_layer=None):
    stem = nn.Sequential()
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    assert len(out_chs)
    in_c = in_chans
    for i, out_c in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        stem.add_module(conv_name, ConvBnAct(
            in_c, out_c, kernel_size, stride=stride if i == 0 else 1,
            act_layer=act_layer, norm_layer=norm_layer))
        in_c = out_c
        last_conv = conv_name
    if pool:
        if aa_layer is not None:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            stem.add_module('aa', aa_layer(channels=in_c, stride=2))
        else:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return stem, dict(num_chs=in_c, reduction=stride, module='.'.join(['stem', last_conv]))


class ResBottleneck(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.25, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_last=False,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResBottleneck, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer,
                       aa_layer=aa_layer, drop_block=drop_block)

        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, mid_chs, kernel_size=3,
                               dilation=dilation, groups=groups, **ckwargs)
        self.attn2 = create_attn(attn_layer, channels=mid_chs) if not attn_last else None
        self.conv3 = ConvBnAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.attn3 = create_attn(attn_layer, channels=out_chs) if attn_last else None
        self.drop_path = drop_path
        self.act3 = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn2 is not None:
            x = self.attn2(x)
        x = self.conv3(x)
        if self.attn3 is not None:
            x = self.attn3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        # FIXME partial shortcut needed if first block handled as per original, not used for my current impl
        #x[:, :shortcut.size(1)] += shortcut
        x = self.act3(x)
        return x


class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer,
                       aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, out_chs, kernel_size=3,
                               dilation=dilation, groups=groups, **ckwargs)
        self.attn = create_attn(attn_layer, channels=out_chs)
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        return x


class CrossStage(nn.Module):
    """Cross Stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1., bottle_ratio=1., exp_ratio=1.,
                 groups=1, first_dilation=None, down_growth=False, cross_linear=False, block_dpr=None,
                 block_fn=ResBottleneck, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs  # grow downsample channels to output channels
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'),
                           norm_layer=block_kwargs.get('norm_layer'))

        if stride != 1 or first_dilation != dilation:
            self.conv_down = ConvBnAct(
                in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
                aa_layer=block_kwargs.get('aa_layer', None), **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs

        # FIXME this 1x1 expansion is pushed down into the cross and block paths in the darknet cfgs. Also,
        # there is also special case for the first stage for some of the model that results in uneven split
        # across the two paths. I did it this way for simplicity for now.
        self.conv_exp = ConvBnAct(prev_chs, exp_chs, kernel_size=1,
                                  apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2  # output of conv_exp is always split in two

        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(
                prev_chs, block_out_chs, dilation, bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition_b = ConvBnAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvBnAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        split = x.shape[1] // 2
        xs, xb = x[:, :split], x[:, split:]
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out


class DarkStage(nn.Module):
    """DarkNet stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1., bottle_ratio=1., groups=1,
                 first_dilation=None, block_fn=ResBottleneck, block_dpr=None, **block_kwargs):
        super(DarkStage, self).__init__()
        first_dilation = first_dilation or dilation

        self.conv_down = ConvBnAct(
            in_chs, out_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups,
            act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'),
            aa_layer=block_kwargs.get('aa_layer', None))

        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(
                prev_chs, block_out_chs, dilation, bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x


def _cfg_to_stage_args(cfg, curr_stride=2, output_stride=32, drop_path_rate=0.):
    # get per stage args for stage and containing blocks, calculate strides to meet target output_stride
    num_stages = len(cfg['depth'])
    if 'groups' not in cfg:
        cfg['groups'] = (1,) * num_stages
    if 'down_growth' in cfg and not isinstance(cfg['down_growth'], (list, tuple)):
        cfg['down_growth'] = (cfg['down_growth'],) * num_stages
    if 'cross_linear' in cfg and not isinstance(cfg['cross_linear'], (list, tuple)):
        cfg['cross_linear'] = (cfg['cross_linear'],) * num_stages
    cfg['block_dpr'] = [None] * num_stages if not drop_path_rate else \
        [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg['depth'])).split(cfg['depth'])]
    stage_strides = []
    stage_dilations = []
    stage_first_dilations = []
    dilation = 1
    for cfg_stride in cfg['stride']:
        stage_first_dilations.append(dilation)
        if curr_stride >= output_stride:
            dilation *= cfg_stride
            stride = 1
        else:
            stride = cfg_stride
            curr_stride *= stride
        stage_strides.append(stride)
        stage_dilations.append(dilation)
    cfg['stride'] = stage_strides
    cfg['dilation'] = stage_dilations
    cfg['first_dilation'] = stage_first_dilations
    stage_args = [dict(zip(cfg.keys(), values)) for values in zip(*cfg.values())]
    return stage_args


class CSPNet(Backbone):
    """Cross Stage Partial base model.

    Paper: `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    Ref Impl: https://github.com/WongKinYiu/CrossStagePartialNetworks

    NOTE: There are differences in the way I handle the 1x1 'expansion' conv in this impl vs the
    darknet impl. I did it this way for simplicity and less special cases.
    """

    def __init__(self, cfg, in_chans=3, output_stride=32, global_pool='avg', drop_rate=0.,
                 act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_path_rate=0.,
                 zero_init_last_bn=True, stage_fn=CrossStage, block_fn=ResBottleneck, out_features=None):
        super().__init__()
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)

        # Construct the stem
        self.stem, stem_feat_info = create_stem(in_chans, **cfg['stem'], **layer_args)
        self.feature_info = [stem_feat_info]
        prev_chs = stem_feat_info['num_chs']
        curr_stride = stem_feat_info['reduction']  # reduction does not include pool
        if cfg['stem']['pool']:
            curr_stride *= 2

        # Construct the stages
        per_stage_args = _cfg_to_stage_args(
            cfg['stage'], curr_stride=curr_stride, output_stride=output_stride, drop_path_rate=drop_path_rate)
        self.stages = nn.Sequential()
        out_channels = []
        out_strides = []
        for i, sa in enumerate(per_stage_args):
            self.stages.add_module(
                str(i), stage_fn(prev_chs, **sa, **layer_args, block_fn=block_fn))
            prev_chs = sa['out_chs']
            curr_stride *= sa['stride']
            self.feature_info += [dict(num_chs=prev_chs,
                                       reduction=curr_stride, module=f'stages.{i}')]
            out_channels.append(prev_chs)
            out_strides.append(curr_stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

        # cspdarknet: csp1, csp2, csp3, csp4
        # cspresnet: csp0, csp1, csp2, csp3
        out_features_names = ["csp{}".format(i) for i in range(len(per_stage_args))]
        self._out_feature_strides = dict(zip(out_features_names, out_strides))
        self._out_feature_channels = dict(zip(out_features_names, out_channels))
        if out_features is None:
            self._out_features = out_features_names
        else:
            self._out_features = out_features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def size_divisibility(self):
        return 32

    def forward(self, x):
        x = self.stem(x)
        outputs = {}
        for i, stage in enumerate(self.stages):
            name = f"csp{i}"
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_cspnet_backbone(cfg, input_shape=None):

    cspnet_name = cfg.MODEL.CSPNET.NAME
    norm_name = cfg.MODEL.CSPNET.NORM
    out_features = cfg.MODEL.CSPNET.OUT_FEATURES
    # DarkNet53 doesn't have batch norm
    if norm_name == "FrozenBN":
        norm = FrozenBatchNorm2d
    elif norm_name == "SyncBN":
        norm = nn.SyncBatchNorm
    else:
        norm = nn.BatchNorm2d

    assert cspnet_name in ["cspresnet50", "cspresnet50d", "cspresnet50w",
                           "cspresnext50", "cspdarknet53", "darknet53"]

    model_cfg = model_cfgs[cspnet_name]

    if "darknet" in cspnet_name:
        block_fn = DarkBlock
    else:
        block_fn = ResBottleneck

    if cspnet_name == "darknet53":
        stage_fn = DarkStage
    else:
        stage_fn = CrossStage

    model = CSPNet(
        model_cfg,
        in_chans=input_shape.channels,
        norm_layer=norm,
        stage_fn=stage_fn,
        block_fn=block_fn,
        out_features=out_features)
    return model
