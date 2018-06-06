# 1. Build docker image

To build the docker image from the Dockerfile located in `dockerfile` please do:
```
cd dockerfile
docker build -t atmyra_docker .
```


Also please make sure that [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) and proper nvidia drivers are installed.

To test the installation run
```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

Then launch the container as follows:
```
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it -v /your/folder/:/home/keras/notebook/your_folder -p 8888:8888 -p 6006:6006 --name atmyra --shm-size 16G atmyra_docker
```

Please note that w/o `--shm-size 16G` PyTorch dataloader classes will not work.
The above command will start a container with a Jupyter notebook server available via port `8888`. 
Port `6006` is for tensorboard, if necessary.

Then you can exec into the container like this. All the scripts were run as root, but they must also work under user `keras`
```
docker exec -it --user root 46b9bd3fa3f8 /bin/bash
```
or
```
docker exec -it --user keras 46b9bd3fa3f8 /bin/bash
```

To find out the container ID run
```
 docker container ls
```

# 2. Train AV's CNNs

## Prepare the data
First `cd av_cnns`


Then make sure that the following files and folders are available via the following relative paths
- `./data/img_list_1M.csv` - list with 1M images
- `../data/img_descriptors_1M.npy` - numpy array with 1M descriptors
- `../data/student_model_imgs` - a folder with 1M images

## Train models

Then copy the following scripts one by one to a `run.sh` file and run `sh run.sh`
To view TensorBoard logs you need to enable TensorBoard via
```
tensorboard --logdir='path/to/av_cnns/tb_logs' --port=6006
```

The weights will be saved to `weights/`
Alternatively you can run all the scripts as one file

**Densenet**
```
python3 distill_network.py \
	--arch densenet161 --lognumber densenet161_1e4_scale \
	--epochs 25 --start-epoch 0 \
	--batch-size 256 --workers 10 \
	--val_size 0.1 --do_augs False \
	--lr 1e-4 --ths 1e-2 \
	--m1 5 --m2 15 \
	--optimizer adam --print-freq 10 \
	--tensorboard True \
```
**ResNet34**
```
python3 distill_network.py \
	--arch resnet18 --lognumber resnet18_scale \
	--epochs 25 --start-epoch 0 \
	--batch-size 512 --workers 10 \
	--val_size 0.1 --do_augs False \
	--lr 1e-3 --ths 1e-2 \
	--m1 5 --m2 15 \
	--optimizer adam --print-freq 10 \
	--tensorboard True \
```

**ResNet18**
```
python3 distill_network.py \
	--arch resnet34 --lognumber resnet34_scale \
	--epochs 25 --start-epoch 0 \
	--batch-size 512 --workers 10 \
	--val_size 0.1 --do_augs False \
	--lr 1e-3 --ths 1e-2 \
	--m1 5 --m2 15 \
	--optimizer adam --print-freq 10 \
	--tensorboard True \
```

## Pre-trained weights
To download the **pre-trained weights** you can use the following links:
- [DenseNet169](https://drive.google.com/open?id=1STT7CIKY8k3k_6RvRX1vop1HEF4A5EEP)
- [ResNet34](https://drive.google.com/open?id=17z5p02kBePmyzyPxdWaHyCb6CVHhMXbe)
- [ResNet18](https://drive.google.com/open?id=1K5zBBxYRzFDqPQqGQ15Lo4vjrexwVtM1)


Also you can add a `-resume` flag to start from a checkpoint:
```
python3 distill_network.py \
	--arch resnet18 --lognumber resnet18_scale \
	--epochs 30 --start-epoch 0 \
	--batch-size 512 --workers 10 \
	--val_size 0.1 --do_augs False \
	--lr 1e-3 --ths 1e-2 \
	--m1 5 --m2 15 \
	--optimizer adam --print-freq 10 \
	--tensorboard True \
	--resume weights/your_weights.pth.tar
```

## Training time
With the above setting on 2x1080Ti training takes:
- 2 hours for ResNet18
- 3 hours for ResNet34
- 11 hours for DenseNet169

![training_curves](av_cnns/training_curves.jpg)


# 3. Train other CNNs



# 4. Run inference attack

run script
```
sh some_script.sh
```