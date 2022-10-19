import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.nn import Conv2d
import mindspore.ops as ops

__all__=["InstanceContextEncoder"]


class PyramidPoolingModule(nn.Cell):
	def __init__(self,in_channels,channels=512,sizes=(1,2,3,6)):
		super().__init__()
		self.stages=[]
		self.stages=nn.CellList([self._make_stage(in_channels,channels,size) for size in sizes])
		self.bottleneck=Conv2d(in_channels+len(sizes)*channels,in_channels,1,has_bias=False)

	def _make_stage(self,features,out_features,size):
		prior=nn.AdaptiveAvgPool2d(output_size=(size,size))
		conv=nn.Conv2d(features,out_features,1,has_bias=True)
		return nn.SequentialCell(prior,conv)

	def construct(self,feats):
		h, w = feats.shape[2], feats.shape[3]

		prior=[ops.ResizeBilinear((h,w))(ops.ReLU()(stage(feats))) for stage in self.stages]+[feats]
		out=ops.ReLU()(self.bottleneck(ops.Concat(axis=1)(prior)))
		return out



class InstanceContextEncoder(nn.Cell):
	def __init__(self,cfg,input_shape):
		super().__init__()
		self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS  #256
		self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES  #[‘res3','res4','res5']
		# self.norm = cfg.MODEL.SPARSE_INST.ENCODER.NORM
		# depthwise = cfg.MODEL.SPARSE_INST.ENCODER.DEPTHWISE
		self.in_channels = [input_shape[f] for f in self.in_features]
		# self.using_bias = self.norm == ""
		fpn_laterals = []
		fpn_outputs = []
		# groups = self.num_channels if depthwise else 1
		for in_channel in reversed(self.in_channels):
			lateral_conv = nn.Conv2d(in_channel, self.num_channels, 1,has_bias=True)
			output_conv = nn.Conv2d(self.num_channels, self.num_channels, 3,has_bias=True)
			fpn_laterals.append(lateral_conv)
			fpn_outputs.append(output_conv)
		self.fpn_laterals = nn.CellList(fpn_laterals)
		self.fpn_outputs = nn.CellList(fpn_outputs)
		# ppm
		self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
		# final fusion
		self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1,has_bias=True)


	def construct(self, features): #features:dict
		features = [features[f] for f in self.in_features]
		features = features[::-1]
		prev_features = self.ppm(self.fpn_laterals[0](features[0]))
		outputs = [self.fpn_outputs[0](prev_features)]

		for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
			lat_features = lat_conv(feature)

			h,w=prev_features.shape[2],prev_features.shape[3]
			top_down_features = ops.ResizeNearestNeighbor(size=(h*2,w*2))(prev_features)###

			prev_features = lat_features + top_down_features
			outputs.insert(0, output_conv(prev_features))

		size = outputs[0].shape[2:]
		features = [outputs[0]] + [ops.ResizeBilinear(size)(x) for x in outputs[1:]]

		features = self.fusion(ops.Concat(axis=1)(features))
		return features
