# NPMT++
Research and development project @ IITB

To run https://github.com/posenhuang/NPMT

Steps followed on MSI GF63 on Jan 23,2020

INSTALL CUDA

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda

HACK REQUIRED

git config --global url.https://github.com/.insteadOf git://github.com/

FOR TORCH INSTALLATION

git clone https://github.com/nagadomi/distro.git ~/torch --recursive

cd ~/torch

./install-deps

./clean.sh

./install.sh

./update.sh

. ~/torch/install/bin/torch-activate

luarocks install nn


Install fairseq by cloning the GitHub repository and running

luarocks make rocks/fairseq-scm-1.rockspec

LuaRocks will fetch and build any additional dependencies that may be missing.
