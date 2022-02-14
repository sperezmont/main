#! /bin/bash
conda_path='/home/sergio/'
env_name='yelmo_tools' # environment name
py_version='3.7.3'       # python version

echo 'Installing requirements ...'
source ${conda_path}/anaconda3/etc/profile.d/conda.sh
conda create -n ${env_name} python=${py_version}  
conda activate ${env_name}
pip install -r yelmo-tools_req.txt
conda deactivate 
echo 'Ready'
