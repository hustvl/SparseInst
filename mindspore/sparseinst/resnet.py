import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from typing import Type, Union, List, Optional
from mindvision.classification.models import ResidualBlock,ResidualBlockBase
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.classification.models.blocks import ConvNormActivation
from collections import OrderedDict


class ResNet(nn.Cell):
	"""
	ResNet architecture.

	Args:
		block (Type[Union[ResidualBlockBase, ResidualBlock]]): THe block for network.
		layer_nums (list): The numbers of block in different layers.
		group (int): The number of Group convolutions. Default: 1.
		base_width (int): The width of per group. Default: 64.
		norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

	Inputs:
		- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
	"""
	def __init__(self,
				block: Type[Union[ResidualBlockBase, ResidualBlock]],
				layer_nums: List[int],
				group: int = 1,
				base_width: int = 64,
				norm: Optional[nn.Cell] = None
				) -> None:
		super(ResNet, self).__init__()
		self.output_shape={}
		if not norm:
			norm = nn.BatchNorm2d
		self.norm = norm
		self.in_channels = 64
		self.group = group
		self.base_with = base_width
		self.stem=OrderedDict()
		conv1 = ConvNormActivation(
			3, self.in_channels, kernel_size=7, stride=2, norm=norm)
		self.stem['conv1']=conv1
		self.stem['maxpool2d']=nn.MaxPool2d(kernel_size=3,stride=2,pad_mode='same')
		self.stem=nn.SequentialCell(self.stem)
		self.res2 = self._make_layer(block, 64, layer_nums[0])
		self.res3 = self._make_layer(block, 128, layer_nums[1], stride=2)
		self.res4 = self._make_layer(block, 256, layer_nums[2], stride=2)
		self.res5 = self._make_layer(block, 512, layer_nums[3], stride=2)

	def _make_layer(self,
					block: Type[Union[ResidualBlockBase, ResidualBlock]],
					channel: int,
					block_nums: int,
					stride: int = 1
					):

		down_sample = None

		if stride != 1 or self.in_channels != self.in_channels * block.expansion:
			down_sample = ConvNormActivation(
				self.in_channels,
				channel * block.expansion,
				kernel_size=1,
				stride=stride,
				norm=self.norm,
				activation=None)
		layers = []
		layers.append(
			block(
				self.in_channels,
				channel,
				stride=stride,
				down_sample=down_sample,
				group=self.group,
				base_width=self.base_with,
				norm=self.norm
			)
		)
		self.in_channels = channel * block.expansion

		for _ in range(1, block_nums):
			layers.append(
				block(
					self.in_channels,
					channel,
					group=self.group,
					base_width=self.base_with,
					norm=self.norm
				)
			)

		return nn.SequentialCell(layers)

	def output_channel(self):
		output_channel={'res3':512,'res4':1024,'res5':2048}
		return output_channel

	def construct(self, x):
		output={}
		x = self.stem(x)

		x = self.res2(x)
		x = self.res3(x)
		output['res3']=x
		x = self.res4(x)
		output['res4']=x
		x = self.res5(x)
		output['res5']=x

		return output



def build_resnet50():
	return ResNet(ResidualBlock,[3,4,6,3])

