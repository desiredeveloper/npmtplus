# NPMT++
Research and development project @ IITB

To run https://github.com/posenhuang/NPMT

I have used anaconda for python environment,

conda create -n py14 python=3.7

conda activate py14

conda install pytorch=1.4 -c pytorch

This will automatically install CUDA and NVCC and other requirements.

Go to CMakeLists.txt and replace

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} - ${OpenMP_CXX_FLAGS}")

with

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES ${OpenMP_CXX_FLAGS}")

luarocks make rocks/fairseq-scm-1.rockspec


