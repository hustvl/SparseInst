import mindspore
import argparse
import numpy as np
from sparseinst import SparseInst, cfg
import cv2
import os
import json
from mindspore import Tensor, ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tqdm
from dict import id2category

def parse_args():
	parser = argparse.ArgumentParser()
    # general
	#parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)
	parser.add_argument('--checkpoint',help="checkpoint path",type=str)
	parser.add_argument('--json_save_path',help="result json scve path",required=False,type=str)
	parser.add_argument("--visualize",action="store_true",help="Run or not.")
	parser.add_argument('--image_name',help="image to visual",required=False,type=str)
	parser.add_argument('--coco_path',help="coco to path",required=False,type=str)
	parser.add_argument('--dir_path',help="coco to visual",required=False,type=str)
	args = parser.parse_args()
	return args


def load_net(path):
	param_dict = mindspore.load_checkpoint(path)
	net = SparseInst(cfg)
	mindspore.load_param_into_net(net, param_dict)
	model=mindspore.Model(network=net)
	return model

def resize_img(img,short_length=640,long_length=864):
	h,w=img.shape[2:]
	image_size=(h,w)

	if h>w:
		h=int(h/w*short_length)
		if h>long_length:
			w=int(w/h*long_length)
			h=long_length
		else:
			w=short_length
	else:
		w=int(w/h*short_length)
		if w>long_length:
			h=int(h/w*long_length)
			w=long_length
		else:
			h=short_length
	img=ops.interpolate(img,sizes=(h,w),mode='bilinear')
	return {'image':img,'image_size':image_size}  ##########

def read_img(name):
	image=cv2.imread(name)
	_image=image.copy()
	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	image=Tensor(image).astype('float32')
	image=ops.transpose(image,(2,0,1))
	image=ops.expand_dims(image,0)
	return image,_image

class Dataset:
	def __init__(self,coco_path,dir_path,short_length=640,long_length=864,visualize=True):
		self.short_length=short_length
		self.long_length=long_length
		self.visualize=visualize
		self.coco = COCO(coco_path)
		self.ids = list(self.coco.imgs.keys())
		self.dir=dir_path

	def __len__(self):
		return len(self.ids)

	def _get_image_path(self, file_name):
		images_dir=self.dir
		return os.path.join(images_dir, file_name)

	def __getitem__(self,index):
		coco=self.coco
		img_id=self.ids[index]
		file_name=coco.loadImgs(img_id)[0]['file_name']
		file_name=self._get_image_path(file_name)
		image,ori_image=read_img(file_name)
		image=resize_img(image)
		return {'image':image,'ori_image':ori_image,'image_id':self.ids[index]}


class Evaluator:
	def __init__(self, coco_path):
		self.coco = COCO(coco_path)

	def evaluate(self, res_file):
		coco_dt = self.coco.loadRes(res_file)
		coco_eval = COCOeval(self.coco, coco_dt, "segm")
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()
		info_str = []
		stats_names = ['AP', 'Ap .5', 'AP .75','AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
		for ind, name in enumerate(stats_names):
			info_str.append((name, coco_eval.stats[ind]))
		return info_str


def read_names(path):
	files = os.listdir(path)
	files=[os.path.join(path,name) for name in files]
	return files

def visualization(masks,image,name,path):
	masks=[mask*255 for mask in masks]
	h,w=masks[0].shape
	path=path+name+'/'
	if not os.path.exists(path):
		os.mkdir(path)
	_=[cv2.imwrite(path+'image_mask'+str(i)+".jpg",((mask.reshape(h,w,1).astype(np.float32)/255.0)*image.astype(np.float32)).astype(np.uint8)) for i,mask in enumerate(masks)]
	_=[cv2.imwrite(path+'mask'+str(i)+'.jpg',mask) for i,mask in enumerate(masks)]

class Runner:
	def __init__(self,dataset,model,visualize=True):
		self.dataset=dataset
		self.model=model
		self.visualize=visualize
		self.dict=list(id2category.keys())
	def __call__(self,idx):
		input=self.dataset[idx]
		ori_image=input['ori_image']
		image_id=input['image_id']
		input=input['image']
		output=self.model.predict(input)[0]['instances']
		if 'pred_masks' in output.keys():
			output['segmentation'] = output['segmentation'].asnumpy()
			if not self.visualize:
				output['']=self.mask2rle(output)
		output['scores'] = output['scores'].asnumpy().astype(float).tolist()
		output['category_id'] = output['category_id'].asnumpy().astype(int).tolist()
		output['category_id']=[self.dict[i] for i in output['category_id']]
		output['image_id']=int(image_id)
		del input
		if not self.visualize:
			del ori_image
			all_pred=[]
			for i,mask in enumerate(output['segmentation']):
				all_pred.append({'image_id':image_id,'category_id':output['category_id'][i],'segmentation':mask,'score':output['scores'][i]})
			del output
			return all_pred #################
		else:
			output['ori_image']=ori_image
			return output
	def mask2rle(self,outputs):
		masks=outputs['pred_masks']
		masks=[mask for mask in masks]
		def f(img):
			'''
			img: numpy array, 1 - mask, 0 - background
			Returns run length as string formated
			'''
			pixels= img.T.flatten()
			pixels = np.concatenate([[0], pixels, [0]])
			runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
			runs[1::2] -= runs[::2]
			return ' '.join(str(x) for x in runs)
		rle=[f(mask) for mask in masks]
		return rle


def main():
	args = parse_args()
	mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
	model=load_net(args.checkpoint)
	if args.visualize:
		image,ori_image=read_img(args.image_name)
		image=resize_img(image)
		dataset=[{'image':image,'ori_image':ori_image,'image_id':args.image_name.split('/')[-1].split('.')[0]}]
	else:
		dataset=Dataset(args.coco_path,args.dir_path,visualize=args.visualize)
	runner=Runner(dataset=dataset,model=model,visualize=args.visualize)
	results=[]
	for i in range(len(dataset)):
		results.append(runner(i))
		print(i)
	if args.visualize:
		_=[visualization(res['pred_masks'],res['ori_image'],res['image_id'],'./') for res in results]
	else:
		res_file=os.path.join(args.json_save_path, "segment_coco_results.json")
		json.dump(results, open(res_file, 'w'))
		eva=Evaluator(args.coco_path)
		info_str=eva.evaluate(res_file)


if __name__=="__main__":
	main()
	