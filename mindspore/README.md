# SparseInst on MindSpore

##  Installation  

1. create python 3.8 environment  
```bash
conda create -n sparseinst-ms python=3.8  
```
2. activate the new environment  
```bash
conda activate sparseinst-ms    
```

3. install mindspore   
``` bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.8.1-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple 
``` 

4. install dependencies   
```bash
pip install mindvision pycocotools opencv-python numpy yacs   
```

##  Model

We provide the basic SparseInst-R50-GIAM in [BaiduPan](https://pan.baidu.com/s/1ZmZ6nqZrwt4ALYP1B2kdCA?pwd=7xsb).

##  Demo

```bash
python test.py --config /path/to/your/checkpoint  --image_name /path/to/your/image --visualize  
```

The results will be saved in ./image_name/
