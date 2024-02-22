conda create --name itcd_env python=3.8
conda activate itcd_env

conda install nvidia::cuda
conda install nvidia::cuda-toolkit

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements.txt