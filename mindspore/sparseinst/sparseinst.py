import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import cv2

from .resnet import build_resnet50
from .encoder import InstanceContextEncoder
from .decoder import GroupIAMDecoder


__all__=["SparseInst"]

def rescoring_mask(scores, mask_pred, masks):
	mask_pred_ = mask_pred.astype('float32')
	return scores * ((masks * mask_pred_).sum(axis=(1, 2)) / (mask_pred_.sum(axis=(1, 2)) + 1e-6))

class SparseInst(nn.Cell):
	def __init__(self,cfg,is_train=False):
		super().__init__()

		self.backbone=build_resnet50()
		self.encoder=InstanceContextEncoder(cfg,self.backbone.output_channel())
		self.decoder=GroupIAMDecoder(cfg)

		self.pixel_mean=Tensor(cfg.MODEL.PIXEL_MEAN).view((3,1,1))
		self.pixel_std=Tensor(cfg.MODEL.PIXEL_STD).view((3,1,1))

		self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
		self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
		self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS

		self.training=is_train

	def normalizer(self, image):
		image=(image-self.pixel_mean)/self.pixel_std
		return image

	def padding(self,image,size_divisibility=32,pad_value=0.0):
		h,w=image.shape[2],image.shape[3]
		bottom=(h//size_divisibility+1)*size_divisibility-h
		right=(w//size_divisibility+1)*size_divisibility-w
		return ops.Pad(((0,0),(0,0),(0,bottom),(0,right)))(image)

	def preprocess_inputs(self,batched_inputs):
		images=self.padding(self.normalizer(batched_inputs))
		return images

	def construct(self,batched_inputs):

			#input :Tensor(N,C,H,W)
			#output = {
			#"pred_logits": pred_logits,
			#"pred_masks": pred_masks,
			#"pred_scores": pred_scores,
		#}
		image_sizes=[batched_inputs['image'].shape[2:]]
		images=self.preprocess_inputs(batched_inputs['image'])
		max_shape=images.shape[2:]
		features=self.backbone(images)
		features=self.encoder(features)
		output=self.decoder(features)
		if self.training:
			return output
		else:
			results=self.inference(output,[batched_inputs],max_shape,image_sizes)
			processed_results=[{'instances':r} for r in results]
			return processed_results


	def inference(self,output,batched_inputs,max_shape,image_sizes):
		results = []
		pred_scores = ops.Sigmoid()(output["pred_logits"])
		pred_masks = ops.Sigmoid()(output["pred_masks"])
		pred_objectness = ops.Sigmoid()(output["pred_scores"])
		pred_scores = ops.Sqrt()(pred_scores * pred_objectness)
		for _, (scores_per_image, mask_pred_per_image, batched_input, img_shape) in enumerate(zip(pred_scores, pred_masks, batched_inputs, image_sizes)):

			labels,scores  = ops.max(scores_per_image,axis=-1)
			keep = scores > self.cls_threshold
			scores = ops.masked_select(scores,keep)
			labels = ops.masked_select(labels,keep)
			n,h,w=mask_pred_per_image.shape
			mask_pred_per_image = ops.masked_select(mask_pred_per_image,keep.view(n,1,1)).view(-1,h,w)
			result={}
			if scores.shape[0]==0:
				result['scores']=scores
				result['category_id']=labels
				results.append(result)
				continue
			h,w=img_shape
			ori_shape=batched_input['image_size']
			scores = rescoring_mask(scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image)
			mask_pred_per_image=ops.interpolate(ops.ExpandDims()(mask_pred_per_image,1),sizes=max_shape,mode='bilinear')
			mask_pred_per_image=mask_pred_per_image.asnumpy()
			mask_pred_per_image=mask_pred_per_image[:,:,:h,:w]
			mask_pred_per_image=Tensor(mask_pred_per_image)

			mask_pred_per_image=ops.interpolate(mask_pred_per_image,sizes=ori_shape,mode='bilinear')
			mask_pred_per_image=ops.squeeze(mask_pred_per_image,axis=1)
			mask_pred=mask_pred_per_image>self.mask_threshold
			mask_pred=mask_pred.astype('uint8')

			result['segmentation'] = mask_pred
			result['scores'] = scores
			result['category_id'] = labels
			results.append(result)

		return results