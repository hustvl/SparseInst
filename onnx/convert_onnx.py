import math
import argparse

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from sparseinst import add_sparse_inst_config


class PyramidPoolingModuleONNX(nn.Module):

    def __init__(self, in_channels, channels, input_size, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, input_size, pool_size)
             for pool_size in pool_sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(pool_sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, input_size, pool_size):
        stride_y = math.floor((input_size[0] / pool_size))
        stride_x = math.floor((input_size[1] / pool_size))
        kernel_y = input_size[0] - (pool_size - 1) * stride_y
        kernel_x = input_size[1] - (pool_size - 1) * stride_x
        prior = nn.AvgPool2d(kernel_size=(
            kernel_y, kernel_x), stride=(stride_y, stride_x))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(
            input=F.relu_(stage(feats)), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


def main():
    parser = argparse.ArgumentParser(
        description="Export model to the onnx format")
    parser.add_argument(
        "--config-file",
        default="configs/sparse_inst_r50_giam.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=640, type=int)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument(
        "--output",
        default="output/sparseinst.onnx",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # norm for ONNX: change FrozenBN back to BN
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "BN"

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger(output=output_dir)
    logger.info(cfg)

    height = args.height
    width = args.width

    model = build_model(cfg)
    num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
    onnx_ppm = PyramidPoolingModuleONNX(
        num_channels, num_channels // 4, (height // 32, width // 32))
    model.encoder.ppm = onnx_ppm
    model.to(cfg.MODEL.DEVICE)
    logger.info("Model:\n{}".format(model))

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    logger.info("load Model:\n{}".format(cfg.MODEL.WEIGHTS))

    input_names = ["input_image"]
    dummy_input = torch.zeros((1, 3, height, width)).to(cfg.MODEL.DEVICE)
    output_names = ["scores", "masks"]

    model.forward = model.forward_test

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=False,
        opset_version=12,
    )

    logger.info("Done. The onnx model is saved into {}.".format(args.output))


if __name__ == "__main__":
    main()
