#### These instructions install pytorch 0.4.0 and NCCL 2.3 built from source using CUDA 8.0.61-1 and cuDNN 7.0.5 on summitdev. <br>
The intructions available at https://developer.ibm.com/tutorials/install-pytorch-on-power/ have been modified specifically for summitdev. 

## Steps:
#### 1. Load required modules and execute some housekeeping commands
```
module load gcc/4.8.5
module load cuda/8.0.61-1

#set encoding to utf-8
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

export CMAKE_PREFIX_PATH="/ccs/home/shubhankar/miniconda3/bin/../"
```
#### 2. Verify NVIDIA CUDA toolkit and driver
To validate the currently installed driver and toolkit execute the following instructions:

Verify and note the CUDA version
```
$ cat /sw/summitdev/cuda/8.0.61-1/version.txt
  CUDA Version 8.0.61
```
Verify and note the driver version installed.
```
$ cat /proc/driver/nvidia/version 
NVRM version: NVIDIA UNIX ppc64le Kernel Module  390.46  Fri Mar 16 21:03:39 PDT 2018
GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-16) (GCC)
```
Finally, verify that NVIDIA toolkit version is matching the CUDA version.
```
-bash-4.2$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:28:28_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```
#### 3. Copy CUDA 8.0 to home directory and set CUDA path in .bashrc
```
cp -r /sw/summitdev/cuda/8.0.61-1 ~/
```
Symlink for a bug fix
```
cd ~/8.0.61-1/nvvm/libdevice
ln -s libdevice.compute_50.10.bc libdevice.10.bc
```
Set CUDA path
```
export PATH="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/bin:$PATH"
export CUDADIR="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1"
export CUDA_HOME="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1"
export CUDA_BIN_PATH="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/bin"
export CUDA_NVCC_EXECUTABLE="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/bin/nvcc"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/extras/CUPTI/lib64"
export NCCL_ROOT_DIR="/autofs/nccs-svm1_home1/$(whoami)/nccl-2.3"
export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
export LD_LIBRARY_PATH=~"NCCL_LIB_DIR:$LD_LIBRARY_PATH"
export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
export CPATH="$NCCL_INCLUDE_DIR:$CPATH"
```
#### 4. Install CUDNN library
Go to NVIDIA cuDNN repository and download [cuDNN v7.0.5 Library for Linux (Power8)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/8.0_20171129/cudnn-8.0-linux-ppc64le-v7) (login is required) to your local machine.
Rsync the tar file from your local machine to summitdev.
```
rsync -avP ~/Downloads/cudnn-8.0-linux-ppc64le-v7.tgz shubhankar@home.ccs.ornl.gov:~/
```
Untar the .tgz file
```
tar -xvzf cudnn-8.0-linux-ppc64le-v.tgz
```
Create a symlink for a bug fix
```
cd cuda/targets/ppc64le-linux/
ln -s lib64 lib
```
Now you will have a cuDNN directory named `cuda`.
Copy cudnn lib into cuda dir
```
cp ~/cuda/targets/ppc64le-linux/include/cudnn.h ~/8.0.61-1/include
cp ~/cuda/targets/ppc64le-linux/lib/libcudnn* ~/8.0.61-1/lib64
chmod a+r ~/cuda/targets/ppc64le-linux/include/cudnn.h ~/8.0.61-1/lib64/libcudnn*
```
#### 5. Install Miniconda 
```
cd ~
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-ppc64le.sh
chmod 744 Miniconda3-latest-Linux-ppc64le.sh
./Miniconda3-latest-Linux-ppc64le.sh
```
Follow the instructions displayed to accept the license and set the installation path. Add the conda install location to PATH in your ~/.bashrc file.
#### 6. Create a virtual environment (optional)
Though this is an optional step, using a virtual environment will help keep your python projects isolated.
The following commands will create an environment named pt_p3.5 with Python 3.5.
```
~/miniconda3/bin/conda create -n pt_p3.5 python=3.5
source ~/miniconda3/bin/activate pt_3.5
```
#### 6. Install openBLAS and Numpy
```
conda install openblas numpy
```
#### 7. Install NCCL 2.3
```
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout f93fe9b
make -j src.build CUDA_HOME=/ccs/home/shubhankar/8.0.61-1

#Install os-agnostic tarball
make pkg.txz.build
tar Jxvf build/pkg/txz/nccl_2.3.5-5+cuda8.0_ppc64le.txz -C ~/
```
Rename to nccl-2.3
```
cd ~
mv nccl_2.3.5-5+cuda8.0_ppc64le nccl-2.3
```
Fix bug
```
cd nccl-2.3
ln -s LICENSE.txt NCCL-SLA.txt
```
Set NCCL path
```
export NCCL_ROOT_DIR="/autofs/nccs-svm1_home1/shubhankar/nccl-2.3/"
export NCCL_LIB_DIR="/autofs/nccs-svm1_home1/shubhankar/nccl-2.3/lib/"
export NCCL_INCLUDE_DIR="/autofs/nccs-svm1_home1/shubhankar/nccl-2.3/include/"
export LD_LIBRARY_PATH="/autofs/nccs-svm1_home1/shubhankar/nccl-2.3/lib:$LD_LIBRARY_PATH"
export CPATH="/autofs/nccs-svm1_home1/shubhankar/nccl-2.3/include:$CPATH"
```
#### 8. Install Magma library
```
cd ~
wget git clone https://github.com/maxhutch/magma.git
cd magma 
```

Copy the make.inc-openblas to make.inc in order to compile with openblas:
```
cp make.inc.openblas make.inc
```
Set the necessary environment variables:
```
export OPENBLASDIR=~/miniconda3/envs/pt_p3.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENBLASDIR/lib
```
Finally, build Magma from the cloned project directory:
```
make -j160
```
Now we can install Magma, the install prefix can be changed if needed. The default prefix is  ~/magma_install. Modify OPENBLASDIR and CUDADIR if needed:
```
make install prefix=~/magma_install
```
Make sure that Magma was installed correctly by checking for libmagma.a, libmagma.so, libmagma_sparse.a and libmagma_sparse.so under ~/magma_install/lib:
```
ls -l ~/magma_install/lib
```
Set Magma path
```
export MAGMA_HOME=/autofs/nccs-svm1_home1/shubhankar/magma_install
export LD_LIBRARY_PATH=~/magma_install/lib:$LD_LIBRARY_PATH
```

#### 9. Install PyTorch from source
Clone the PyTorch project:
```
cd ~
git clone --recursive https://github.com/pytorch/pytorch ~/pytorch
cd pytorch
git checkout d38adfe35db9eabb500cd1213d34d0526e364a46
```
Install pip dependencies
```
pip install certifi cffi numpy setuptools wheel pip pyyaml
```
python setup.py bdist_wheel
```
Install PyTorch with pip:
```
cd dist
pip install torch-0.4.0a0+d38adfe-cp35-cp35m-linux_ppc64le.whl
```
#### 10. Test the installation
```
cd ~
python -c "import torch; print(torch.__version__);"

