import sys
import typing
from typing import Dict, List, Counter, Any
import logging
import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str
from fvcore.nn.jit_handles import conv_flop_jit, Handle, get_shape, conv_flop_count
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
    TracingAdapter
)

from detectron2.utils.logger import setup_logger
sys.path.append(".")
from sparseinst import add_sparse_inst_config

logger = logging.getLogger("detectron2")


def dconv_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    Count flops for convolution.
    """
    # Inputs of Convolution should be a list of length 12 or 13. They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    # starting with #40737 it will be 12) user_enabled_tf32
    # assert len(inputs) == 12 or len(inputs) == 13, len(inputs)
    x, _, w = inputs[:3]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    return Counter({"conv": conv_flop_count(x_shape, w_shape, out_shape)})


_NEW_SUPPORTED_OPS: Dict[str, Handle] = {
    "prim::PythonOp._DeformConv": dconv_flop_jit,
}


class MyFlopCountAnalysis(FlopCountAnalysis):
    """
    Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models.
    """

    def __init__(self, model, inputs):
        """
        Args:
            model (nn.Module):
            inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
        """
        wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
        super().__init__(wrapper, wrapper.flattened_inputs)
        self.set_op_handle(**_NEW_SUPPORTED_OPS)


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_sparse_inst_config(cfg)
        cfg.merge_from_file(args.config_file)
        print(cfg.MODEL.WEIGHTS)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


def do_flop(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        flops = MyFlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())
        # print(flops.unsupported_ops())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )


def do_activation(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )


def do_parameter(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Model Structure:\n" + str(model))


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
Examples:

To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:

$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1
    cfg = setup(args)
    
    for task in args.tasks:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
        }[task](cfg)
