# Build docker image

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

# Train AV's CNNs



# Train other CNNs



# Run inference attack

run script
```
sh some_script.sh
```