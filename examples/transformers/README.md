### Overview
DLPy relies on the [HuggingFace](https://huggingface.co/transformers/installation.html) implementation of BERT.
You must install the necessary packages to enable HuggingFace support. 

### Sample Installation (assumes Anaconda)
- Create conda environment for transformers (*conda create -n transformers python=X.X.X anaconda*)
- Install PyTorch and dependencies 
    - CPU version: *conda install -n transformers pytorch-cpu torchvision-cpu -c pytorch*
    - GPU version: *conda install pytorch torchvision cudatoolkit=10.1 -c pytorch*
- install Transformers package from HuggingFace (*conda install -n transformers -c conda-forge transformers*)
