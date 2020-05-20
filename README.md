# NPMT++
Research and development project @ IITB

To run https://github.com/posenhuang/NPMT

## Docker environment setup:
```
sudo apt-get remove docker docker-engine docker.io
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
docker --version

sudo groupadd docker
sudo usermod -aG docker ${USER}
sudo chmod 666 /var/run/docker.sock
```
## Pull nvidia-cuda image
```
docker pull nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
```
## Leverage GPU support inside docker containers
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
## Create a docker container 
```
docker run --name npmt --gpus all -t 5aafb863776b bash
```
where 5aafb863776b is image id of docker image pulled

## Start the container and run the following after getting into the container
```
docker exec -it npmt bash

apt-get update
apt-get install git
apt-get -y install sudo

git clone https://github.com/NVIDIA/nccl.git
make CUDA_HOME=/usr/local/cuda -j"$(nproc)"
export LD_LIBRARY_PATH=/root/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mv nccl/build/lib/libnccl.so.2 nccl/build/lib/libnccl.so.1
cd nccl/build
make install
luarocks install nccl
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"

git clone https://github.com/soumith/cudnn.torch
cd cudnn.torch/
git checkout origin/R7
luarocks make cudnn-scm-1.rockspec

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh 

git clone https://github.com/posenhuang/NPMT.git
```


