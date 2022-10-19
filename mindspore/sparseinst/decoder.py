import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import Conv2d


__all__=["BaseIAMDecoder","GroupIAMDecoder"]

def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
	convs = []
	for _ in range(num_convs):
		convs.append(
			nn.Conv2d(in_channels, out_channels, 3, has_bias=True))
		convs.append(nn.ReLU())
		in_channels = out_channels
	return nn.SequentialCell(*convs)


class MaskBranch(nn.Cell):

	def __init__(self, cfg, in_channels):
		super().__init__()
		dim = cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM#256
		num_convs = cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS
		kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
		self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
		self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1,has_bias=True)

	def construct(self, features):
 		# mask features (x4 convs)
		features = self.mask_convs(features)
		return self.projection(features)



class InstanceBranch(nn.Cell):

	def __init__(self, cfg, in_channels):
		super().__init__()
		# norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
		dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
		num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
		num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
		kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
		self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

		self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
		# iam prediction, a simple conv
		self.iam_conv = nn.Conv2d(dim, num_masks, 3, has_bias=True)

		 # outputs
		self.cls_score = nn.Dense(dim, self.num_classes)
		self.mask_kernel = nn.Dense(dim, kernel_dim)
		self.objectness = nn.Dense(dim, 1)


	def construct(self, features):
		# instance features (x4 convs)
		features = self.inst_convs(features)
		# predict instance activation maps
		iam = self.iam_conv(features)
		iam_prob = ops.Sigmoid()(iam)

		B, N = iam_prob.shape[:2]
		C = features.shape[1]
		# BxNxHxW -> BxNx(HW)
		iam_prob = iam_prob.view((B, N, -1))
		# aggregate features: BxCxHxW -> Bx(HW)xC
		inst_features=ops.BatchMatMul(transpose_b=True)(iam_prob,features.view((B, C, -1)))
		normalizer = ops.clip_by_value(iam_prob.sum(-1),clip_value_min=Tensor(1e-6,mindspore.float32))
		inst_features = inst_features / normalizer[:, :, None]
		# predict classification & segmentation kernel & objectness
		pred_logits = self.cls_score(inst_features)
		pred_kernel = self.mask_kernel(inst_features)
		pred_scores = self.objectness(inst_features)
		return pred_logits, pred_kernel, pred_scores, iam


class BaseIAMDecoder(nn.Cell):

	def __init__(self, cfg):
		super().__init__()
		# add 2 for coordinates
		in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2

		self.scale_factor = cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR
		self.output_iam = cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM

		self.resize=nn.ResizeBilinear()
		
		self.inst_branch = InstanceBranch(cfg, in_channels)
		self.mask_branch = MaskBranch(cfg, in_channels)


	def compute_coordinates(self, x):
		h, w = x.shape[2], x.shape[3]
		start=Tensor(-1,mindspore.float32)
		stop=Tensor(1,mindspore.float32)
		y_loc = ops.linspace(start,stop, h)
		x_loc = ops.linspace(start,stop, w)
		y_loc, x_loc = ops.meshgrid((y_loc, x_loc),indexing='ij')
		y_loc=ops.broadcast_to(y_loc,(x.shape[0],1,-1,-1))
		x_loc=ops.broadcast_to(x_loc,(x.shape[0],1,-1,-1))
		locations=ops.concat((x_loc,y_loc),axis=1)
		return locations.astype('float32')
    
	def construct(self, features):
		coord_features = self.compute_coordinates(features)
		features=ops.concat((coord_features,features),axis=1)
		pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
		mask_features = self.mask_branch(features)

		N = pred_kernel.shape[1]
		# mask_features: BxCxHxW
		B, C, H, W = mask_features.shape
		pred_masks=ops.BatchMatMul()(pred_kernel,mask_features.view((B,C,H*W))).view((B,N,H,W))


		pred_masks=self.resize(pred_masks,scale_factor=self.scale_factor)
		output = {
			"pred_logits": pred_logits,
			"pred_masks": pred_masks,
			"pred_scores": pred_scores,
		}

		if self.output_iam:
			iam=self.resize(iam,scale_factor=self.scale_factor)
			output['pred_iam'] = iam

		return output



class GroupInstanceBranch(nn.Cell):

	def __init__(self, cfg, in_channels):
		super().__init__()
		# norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
		dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
		num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
		num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
		kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
		self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES
		self.num_groups = cfg.MODEL.SPARSE_INST.DECODER.GROUPS

		self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
		# iam prediction, a simple conv
		expand_dim = dim * self.num_groups
		self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, group=self.num_groups,has_bias=True)

		 # outputs
		self.fc = nn.Dense(expand_dim, expand_dim)
		self.cls_score = nn.Dense(expand_dim, self.num_classes)
		self.mask_kernel = nn.Dense(expand_dim, kernel_dim)
		self.objectness = nn.Dense(expand_dim, 1)


	def construct(self, features):
		# instance features (x4 convs)
		features = self.inst_convs(features)
		# predict instance activation maps
		iam = self.iam_conv(features)
		iam_prob = ops.Sigmoid()(iam)

		B, N = iam_prob.shape[:2]
		C = features.shape[1]
		# BxNxHxW -> BxNx(HW)
		iam_prob = iam_prob.view((B, N, -1))
		# aggregate features: BxCxHxW -> Bx(HW)xC
		inst_features=ops.BatchMatMul(transpose_b=True)(iam_prob,features.view((B, C, -1)))
		normalizer = ops.clip_by_value(iam_prob.sum(-1),clip_value_min=Tensor(1e-6,mindspore.float32))
		inst_features = inst_features / normalizer[:, :, None]

		inst_features=ops.reshape(ops.Transpose()(ops.reshape(inst_features,(B,4,N//4,-1)),(0,2,1,3)),(B,N//4,-1))
		inst_features=ops.ReLU()(self.fc(inst_features))
		# predict classification & segmentation kernel & objectness
		pred_logits = self.cls_score(inst_features)
		pred_kernel = self.mask_kernel(inst_features)
		pred_scores = self.objectness(inst_features)
		return pred_logits, pred_kernel, pred_scores, iam



class GroupIAMDecoder(BaseIAMDecoder):
	def __init__(self, cfg):
		super().__init__(cfg)
		in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
		self.inst_branch = GroupInstanceBranch(cfg, in_channels)


