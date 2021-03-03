<!--
 * @Author: yuan
 * @Date: 2021-02-20 09:18:49
 * @LastEditTime: 2021-03-03 11:57:37
 * @FilePath: /classification/image-classification/README.md
-->
Custom from [Pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
## Build by docker
```
CMD ["/workspace/yuan-algorithm/image-classification/run.sh"]
```
if you just train , comment out the above line
```bash
$ cd docker/algo
build.sh
$ cd docker/database
build.sh

``` 
## RUN container 
- docker-Compose
```bash
docker-compose up -d
```
## Note
- please change all path depend on your requires (including config.json , docker-compose.yml)



## Usage
### 1. enter directory
```bash
$ cd image-classification
```
### 2. dataset 
#### Train(So far , only support NG/OK two class training)
- train 
  - NG
  - OK
- val 
  - NG
  - OK
####  Predict
You can use datasets that you want to predict classification , and put follow 
- share (depend on your config.json-> "mount_folder_path")
  - job1
    - layer
      - lot 
        - pannel id
  
### 3. run tensorbard(optional)
Install tensorboard
```bash
$ pip install tensorboard
$ mkdir runs
$ cd runs/net_name/
Run tensorboard
$ tensorboard --logdir='Monday_01_February_2021_08h_35m_02s' [--port=6006 --host='localhost']
```

### 4. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train xception by btach size 4 ,
# learning rate 0.003 ,optimizer adam
$ python3 train.py -net xception -gpu -b 4 -lr 0.003 -optim adam 
```

The supported net args are:
```
squeezenet
mobilenet
mobilenetv2
shufflenet
shufflenetv2
vgg11
vgg13
vgg16
vgg19
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
nasnet
wideresnet
stochasticdepth18
stochasticdepth34
stochasticdepth50
stochasticdepth101
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).



### 5. test the model
Test the model using test.py
```bash
$ python3 test.py -net xception -weights ./checkpoint/xception/Monday_01_February_2021_08h_35m_02s/xception-145-best.pth \
                 -gpu -b 4 -s -src datasets/val -des result_xception  
```
## Training Details
I use xavier init [default],no bias decay [default], warmup training [optional], label smoothing [optional]