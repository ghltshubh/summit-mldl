#! /usr/bin/bash

cd ~
mkdir summit
cd summit

# user input for python and pytorch version
read -p "Enter python version (2 or 3): " python_ver
while [ "$python_ver" != 2 ] && [ "$python_ver" != 3 ]; do
	read -p "Enter python version (2 or 3): " python_ver 
done
read -p "Enter pytorch version to install (0.4 or 1.0): " pt_ver
while [ "$pt_ver" != 0.4 ] && [ "$pt_ver" != 1.0 ]; do
	read -p "Enter pytorch version to install (0.4 or 1.0): " pt_ver
done

echo "Installing pytorch $pt_ver with python $python_ver"

CONDA_VER=5.2.0
NCCL_VER=2.3.7-1
## download cuDNN (make sure the link is login free)
#cd ~
#wget https://code.ornl.gov/summitdev/mldl-stack/pytorch/raw/master/dependencies/cuda.tar.gz
#tar -xvzf ~/cuda.tar.gz

## copy cudnn lib into cuda dir
#echo copying cudnn.h ...
#cp ~/cuda/targets/ppc64le-linux/include/cudnn.h ~/9.2.148/include
#echo copying libcudnn* ...
#cp ~/cuda/targets/ppc64le-linux/lib/libcudnn* ~/9.2.148/lib64
#echo changing permissions to a+r for libcudnn*
#chmod a+r ~/cuda/targets/ppc64le-linux/include/cudnn.h ~/9.2.148/lib64/libcudnn*

# create virtual env
# conda create -n myenv

# env() { source $1/bin/activate myenv; }

# env anaconda3

# install pytorch

if [ "$pt_ver" == 1.0 ] && [ "$python_ver" == 3 ]; then
	mkdir pytorch-"$pt_ver"-p"$python_ver"
	cd pytorch-"$pt_ver"-p"$python_ver"
	wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh
        chmod +x anaconda3.sh
	./anaconda3.sh -b -p anaconda3
        export PATH="$(pwd)/anaconda3/bin:$PATH"
        pip install --upgrade pip
        conda install -y keras-applications --no-deps
        conda install -y keras-preprocessing --no-deps
	cp -r /sw/summit/cuda/9.2.148/ $(pwd)/
	CUDA_DIR="$(pwd)/9.2.148"

        #workaround for nccl2 
        git clone https://github.com/NVIDIA/nccl
        cd nccl
        git checkout v$NCCL_VER
        make -j160 src.build CUDA_HOME=$CUDA_DIR
        make pkg.txz.build CUDA_HOME=$CUDA_DIR
        tar -xf  build/pkg/txz/* -C ..
        cd ..
        ln -s nccl_$NCCL_VER* nccl2
        cd nccl2
        ln -s LICENSE.txt NCCL-SLA.txt
        cd ..

        echo 'export LANGUAGE=en_US.UTF-8
        export LC_ALL=en_US.UTF-8
        export LANG=en_US.UTF-8
        export LC_TYPE=en_US.UTF-8
	export PATH="$(pwd)/anaconda3/bin:$PATH"
        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
        export PATH="$(pwd)/9.2.148/bin:$PATH"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
        export CUDADIR="$(pwd)/9.2.148"
        export OPENBLASDIR="$(pwd)/anaconda3"
        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
        export CUDA_HOME="$(pwd)/9.2.148"
        export NCCL_ROOT_DIR="$(pwd)/nccl2"
        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> $(pwd)/source_to_run_pytorch"$pt_ver"-p"$python_ver"

        export LANGUAGE=en_US.UTF-8
        export LC_ALL=en_US.UTF-8
        export LANG=en_US.UTF-8
        export LC_TYPE=en_US.UTF-8
        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
        export PATH="$(pwd)/9.2.148/bin:$PATH"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
        export CUDADIR="$(pwd)/9.2.148"
        export OPENBLASDIR="$(pwd)/anaconda3"
        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
        export CUDA_HOME="$(pwd)/9.2.148"
        export NCCL_ROOT_DIR="$(pwd)/nccl2"
        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"

	wget https://code.ornl.gov/summit/mldl-stack/pytorch/raw/master/wheels/torch-1.0.0a0+ba0ebe3-cp36-cp36m-linux_ppc64le.whl
	
	#numpy fix 
        pip install -U numpy

	pip install torch-1.0.0a0+ba0ebe3-cp36-cp36m-linux_ppc64le.whl
	rm torch-1.0.0a0+ba0ebe3-cp36-cp36m-linux_ppc64le.whl

elif [ "$pt_ver" == 0.4 ] && [ "$python_ver" == 3 ]; then
	mkdir pytorch-"$pt_ver"-p"$python_ver"
        cd pytorch-"$pt_ver"-p"$python_ver"
	wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh
	chmod +x anaconda3.sh
	./anaconda3.sh -b -p anaconda3
	export PATH="$(pwd)/anaconda3/bin:$PATH"
	pip install --upgrade pip
	conda install -y keras-applications --no-deps
	conda install -y keras-preprocessing --no-deps
	cp -r /sw/summit/cuda/9.2.148/ $(pwd)/	
        CUDA_DIR="$(pwd)/9.2.148"

        #workaround for nccl2 
        git clone https://github.com/NVIDIA/nccl
        cd nccl
        git checkout v$NCCL_VER
        make -j160 src.build CUDA_HOME=$CUDA_DIR
        make pkg.txz.build CUDA_HOME=$CUDA_DIR
        tar -xf  build/pkg/txz/* -C ..
        cd ..
        ln -s nccl_$NCCL_VER* nccl2
        cd nccl2
        ln -s LICENSE.txt NCCL-SLA.txt
        cd ..

        echo 'export LANGUAGE=en_US.UTF-8
        export LC_ALL=en_US.UTF-8
        export LANG=en_US.UTF-8
        export LC_TYPE=en_US.UTF-8
	export PATH="$(pwd)/anaconda3/bin:$PATH"
        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
        export PATH="$(pwd)/9.2.148/bin:$PATH"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
        export CUDADIR="$(pwd)/9.2.148"
        export OPENBLASDIR="$(pwd)/anaconda3"
        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
        export CUDA_HOME="$(pwd)/9.2.148"
        export NCCL_ROOT_DIR="$(pwd)/nccl2"
        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> $(pwd)/source_to_run_pytorch"$pt_ver"-p"$python_ver"

        export LANGUAGE=en_US.UTF-8
        export LC_ALL=en_US.UTF-8
        export LANG=en_US.UTF-8
        export LC_TYPE=en_US.UTF-8
        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
        export PATH="$(pwd)/9.2.148/bin:$PATH"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
        export CUDADIR="$(pwd)/9.2.148"
        export OPENBLASDIR="$(pwd)/anaconda3"
        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
        export CUDA_HOME="$(pwd)/9.2.148"
        export NCCL_ROOT_DIR="$(pwd)/nccl2"
        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"

	wget https://code.ornl.gov/summit/mldl-stack/pytorch/raw/master/wheels/torch-0.4.0a0+3749c58-cp36-cp36m-linux_ppc64le.whl

        #numpy fix 
        pip install -U numpy
        pip install torch-0.4.0a0+3749c58-cp36-cp36m-linux_ppc64le.whl
	rm torch-0.4.0a0+3749c58-cp36-cp36m-linux_ppc64le.whl
#elif [ "$pt_ver" == 1.12 ] && [ "$python_ver" == 3 ]; then
#	mkdir pytorch-"$pt_ver"-p"$python_ver"
#        cd pytorch-"$pt_ver"-p"$python_ver"
#        wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh
#        chmod +x anaconda3.sh
#	./anaconda3.sh -b -p anaconda3
#        export PATH="$(pwd)/anaconda3/bin:$PATH"
#        pip install --upgrade pip
#        conda install -y keras-applications --no-deps
#        conda install -y keras-preprocessing --no-deps
#	cp -r /sw/summit/cuda/9.2.148/ $(pwd)/
#        CUDA_DIR="$(pwd)/9.2.148"
#
#        #workaround for nccl2 
#        git clone https://github.com/NVIDIA/nccl
#        cd nccl
#        git checkout v$NCCL_VER
#        make -j160 src.build CUDA_HOME=$CUDA_DIR
#        make pkg.txz.build CUDA_HOME=$CUDA_DIR
#        tar -xf  build/pkg/txz/* -C ..
#        cd ..
#        ln -s nccl_$NCCL_VER* nccl2
#        cd nccl2
#        ln -s LICENSE.txt NCCL-SLA.txt
#        cd ..
#
#        echo 'export LANGUAGE=en_US.UTF-8
#        export LC_ALL=en_US.UTF-8
#        export LANG=en_US.UTF-8
#        export LC_TYPE=en_US.UTF-8
#	export PATH="$(pwd)/anaconda3/bin:$PATH"
#        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
#        export PATH="$(pwd)/9.2.148/bin:$PATH"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
#        export CUDADIR="$(pwd)/9.2.148"
#        export OPENBLASDIR="$(pwd)/anaconda3"
#        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
#        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
#        export CUDA_HOME="$(pwd)/9.2.148"
#        export NCCL_ROOT_DIR="$(pwd)/nccl2"
#        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
#        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
#        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
#        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
#        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> $(pwd)/source_to_run_pytorch"$pt_ver"-p"$python_ver"
#
#        export LANGUAGE=en_US.UTF-8
#        export LC_ALL=en_US.UTF-8
#        export LANG=en_US.UTF-8
#        export LC_TYPE=en_US.UTF-8
#        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
#        export PATH="$(pwd)/9.2.148/bin:$PATH"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
#        export CUDADIR="$(pwd)/9.2.148"
#        export OPENBLASDIR="$(pwd)/anaconda3"
#        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
#        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
#        export CUDA_HOME="$(pwd)/9.2.148"
#        export NCCL_ROOT_DIR="$(pwd)/nccl2"
#        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
#        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
#        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
#        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
#        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"
#
#        wget https://code.ornl.gov/summit/mldl-stack/pytorch/raw/master/wheels/pytorch-1.12.0-cp36-cp36m-linux_ppc64le.whl
#	pip install pytorch-1.12.0-cp36-cp36m-linux_ppc64le.whl
#	rm pytorch-1.12.0-cp36-cp36m-linux_ppc64le.whl
#elif [ "$pt_ver" == 1.12 ] && [ "$python_ver" == 2 ]; then
#	mkdir pytorch-"$pt_ver"-p"$python_ver"
#        cd pytorch-"$pt_ver"-p"$python_ver"
#        wget https://repo.anaconda.com/archive/Anaconda2-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh
#        chmod +x anaconda3.sh
#	./anaconda3.sh -b -p anaconda3
#        export PATH="$(pwd)/anaconda3/bin:$PATH"
#        pip install --upgrade pip
#        conda install -y keras-applications --no-deps
#        conda install -y keras-preprocessing --no-deps
#	cp -r /sw/summit/cuda/9.2.148/ $(pwd)/	
#        CUDA_DIR="$(pwd)/9.2.148"
#
#        #workaround for nccl2 
#        git clone https://github.com/NVIDIA/nccl
#        cd nccl
#        git checkout v$NCCL_VER
#        make -j160 src.build CUDA_HOME=$CUDA_DIR
#        make pkg.txz.build CUDA_HOME=$CUDA_DIR
#        tar -xf  build/pkg/txz/* -C ..
#        cd ..
#        ln -s nccl_$NCCL_VER* nccl2
#        cd nccl2
#        ln -s LICENSE.txt NCCL-SLA.txt
#        cd ..
#
#        echo 'export LANGUAGE=en_US.UTF-8
#        export LC_ALL=en_US.UTF-8
#        export LANG=en_US.UTF-8
#        export LC_TYPE=en_US.UTF-8
#	export PATH="$(pwd)/anaconda3/bin:$PATH"
#        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
#        export PATH="$(pwd)/9.2.148/bin:$PATH"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
#        export CUDADIR="$(pwd)/9.2.148"
#        export OPENBLASDIR="$(pwd)/anaconda3"
#        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
#        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
#        export CUDA_HOME="$(pwd)/9.2.148"
#        export NCCL_ROOT_DIR="$(pwd)/nccl2"
#        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
#        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
#        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
#        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
#        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> $(pwd)/source_to_run_pytorch"$pt_ver"-p"$python_ver"
#
#        export LANGUAGE=en_US.UTF-8
#        export LC_ALL=en_US.UTF-8
#        export LANG=en_US.UTF-8
#        export LC_TYPE=en_US.UTF-8
#        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
#        export PATH="$(pwd)/9.2.148/bin:$PATH"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
#        export CUDADIR="$(pwd)/9.2.148"
#        export OPENBLASDIR="$(pwd)/anaconda3"
#        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
#        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
#        export CUDA_HOME="$(pwd)/9.2.148"
#        export NCCL_ROOT_DIR="$(pwd)/nccl2"
#        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
#        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
#        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
#        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
#        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"
#
#        wget https://code.ornl.gov/summit/mldl-stack/pytorch/raw/master/wheels/pytorch-1.12.0-cp27-cp27mu-linux_ppc64le.whl
#        pip install pytorch-1.12.0-cp27-cp27mu-linux_ppc64le.whl
#	rm pytorch-1.12.0-cp27-cp27mu-linux_ppc64le.whl
#elif [ "$pt_ver" == 1.11 ] && [ "$python_ver" == 2 ]; then
#	mkdir pytorch-"$pt_ver"-p"$python_ver"
#        cd pytorch-"$pt_ver"-p"$python_ver"
#        wget https://repo.anaconda.com/archive/Anaconda2-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh
#        chmod +x anaconda3.sh
#	./anaconda3.sh -b -p anaconda3
#        export PATH="$(pwd)/anaconda3/bin:$PATH"
#        pip install --upgrade pip
#        conda install -y keras-applications --no-deps
#        conda install -y keras-preprocessing --no-deps
#	cp -r /sw/summit/cuda/9.2.148/ $(pwd)/
#        CUDA_DIR="$(pwd)/9.2.148"
#
#        #workaround for nccl2 
#        git clone https://github.com/NVIDIA/nccl
#        cd nccl
#        git checkout v$NCCL_VER
#        make -j160 src.build CUDA_HOME=$CUDA_DIR
#        make pkg.txz.build CUDA_HOME=$CUDA_DIR
#        tar -xf  build/pkg/txz/* -C ..
#        cd ..
#        ln -s nccl_$NCCL_VER* nccl2
#        cd nccl2
#        ln -s LICENSE.txt NCCL-SLA.txt
#        cd ..
#
#        echo 'export LANGUAGE=en_US.UTF-8
#        export LC_ALL=en_US.UTF-8
#        export LANG=en_US.UTF-8
#        export LC_TYPE=en_US.UTF-8
#	export PATH="$(pwd)/anaconda3/bin:$PATH"
#        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
#        export PATH="$(pwd)/9.2.148/bin:$PATH"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
#        export CUDADIR="$(pwd)/9.2.148"
#        export OPENBLASDIR="$(pwd)/anaconda3"
#        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
#        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
#        export CUDA_HOME="$(pwd)/9.2.148"
#        export NCCL_ROOT_DIR="$(pwd)/nccl2"
#        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
#        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
#        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
#        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
#        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> $(pwd)/source_to_run_pytorch"$pt_ver"-p"$python_ver"
#
#        export LANGUAGE=en_US.UTF-8
#        export LC_ALL=en_US.UTF-8
#        export LANG=en_US.UTF-8
#        export LC_TYPE=en_US.UTF-8
#        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
#        export PATH="$(pwd)/9.2.148/bin:$PATH"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
#        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
#        export CUDADIR="$(pwd)/9.2.148"
#        export OPENBLASDIR="$(pwd)/anaconda3"
#        export CMAKE_PREFIX_PATH="$(pwd)/anaconda3/bin/../"
#        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
#        export CUDA_HOME="$(pwd)/9.2.148"
#        export NCCL_ROOT_DIR="$(pwd)/nccl2"
#        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
#        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
#        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
#        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
#        export CPATH="$NCCL_INCLUDE_DIR:$CPATH"
#
#        wget https://code.ornl.gov/summit/mldl-stack/pytorch/raw/master/wheels/pytorch-1.11.0-cp27-cp27mu-linux_ppc64le.whl
#        pip install pytorch-1.11.0-cp27-cp27mu-linux_ppc64le.whl
#	rm pytorch-1.11.0-cp27-cp27mu-linux_ppc64le.whl
elif [ "$pt_ver" == 0.4 ] && [ "$python_ver" == 2 ]; then
	mkdir pytorch-"$pt_ver"-p"$python_ver"
        cd pytorch-"$pt_ver"-p"$python_ver"
        wget https://repo.anaconda.com/archive/Anaconda2-$CONDA_VER-Linux-ppc64le.sh -O anaconda2.sh
        chmod +x anaconda2.sh
	./anaconda2.sh -b -p anaconda2
        export PATH="$(pwd)/anaconda2/bin:$PATH"
        pip install --upgrade pip
        conda install -y keras-applications --no-deps
        conda install -y keras-preprocessing --no-deps
        cp -r /sw/summit/cuda/9.2.148/ $(pwd)/
	CUDA_DIR="$(pwd)/9.2.148"
	
	#workaround for nccl2 
	git clone https://github.com/NVIDIA/nccl
	cd nccl
	git checkout v$NCCL_VER
	make -j160 src.build CUDA_HOME=$CUDA_DIR
	make pkg.txz.build CUDA_HOME=$CUDA_DIR
	tar -xf  build/pkg/txz/* -C ..
	cd ..
	ln -s nccl_$NCCL_VER* nccl2
	cd nccl2
	ln -s LICENSE.txt NCCL-SLA.txt
	cd ..

	echo 'export LANGUAGE=en_US.UTF-8
	export LC_ALL=en_US.UTF-8
	export LANG=en_US.UTF-8
	export LC_TYPE=en_US.UTF-8
	export PATH="$(pwd)/anaconda2/bin:$PATH"
	export PATH="$(pwd)/anaconda2/bin:$PATH"
	export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
	export PATH="$(pwd)/9.2.148/bin:$PATH"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
	export CUDADIR="$(pwd)/9.2.148"
	export OPENBLASDIR="$(pwd)/anaconda2"
	export CMAKE_PREFIX_PATH="$(pwd)/anaconda2/bin/../"
	export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
	export CUDA_HOME="$(pwd)/9.2.148"
	export NCCL_ROOT_DIR="$(pwd)/nccl2"
	export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
	export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
	export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
	export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
	export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> $(pwd)/source_to_run_pytorch"$pt_ver"-p"$python_ver"
	
	export LANGUAGE=en_US.UTF-8
	export LC_ALL=en_US.UTF-8
	export LANG=en_US.UTF-8
	export LC_TYPE=en_US.UTF-8
        export HOROVOD_NCCL_HOME="$(pwd)/nccl2/"
        export PATH="$(pwd)/9.2.148/bin:$PATH"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/lib64"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/9.2.148/extras/CUPTI/lib64"
        export CUDADIR="$(pwd)/9.2.148"
        export OPENBLASDIR="$(pwd)/anaconda2"
        export CMAKE_PREFIX_PATH="$(pwd)/anaconda2/bin/../"
        export CUDA_BIN_PATH="$(pwd)/9.2.148/bin"
        export CUDA_HOME="$(pwd)/9.2.148"
        export NCCL_ROOT_DIR="$(pwd)/nccl2"
        export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
        export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"
        export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
        export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
	export CPATH="$NCCL_INCLUDE_DIR:$CPATH"

	wget https://code.ornl.gov/summit/mldl-stack/pytorch/raw/master/wheels/torch-0.4.0a0+d38adfe-cp27-cp27mu-linux_ppc64le.whl

        #numpy fix 
        pip install -U numpy
        pip install torch-0.4.0a0+d38adfe-cp27-cp27mu-linux_ppc64le.whl
	rm torch-0.4.0a0+d38adfe-cp27-cp27mu-linux_ppc64le.whl
fi
echo -e '\n'
echo -e '\tPytorch successfully installed!!'

