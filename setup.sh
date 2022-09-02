mamba create -n vl -y python=3.8 &&
conda activate vl &&
mamba install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y &&
mamba install gdown tensorboard -c conda-forge -y &&
gdown --fuzzy https://drive.google.com/file/d/1A4DwFb8Jj-PCFcKZE_SJA4b_oN5t9XSm/view?usp=sharing &&
tar -xvf IsaacGym_Preview_3_Package.tar.gz &&
rm IsaacGym_Preview_3_Package.tar.gz &&
pip install -e isaacgym/python &&
pip install -e rsl_rl/. &&
pip install -e legged_gym/. &&
pip install pytorch-memlab

# Habitat Lab installs
pip install typing-extensions~=3.7.4 google-auth==1.6.3 simplejson braceexpand pybullet &&
pip install -r requirements.txt &&
python setup.py develop --all &&
conda install -c conda-forge bullet -y &&
pip install squaternion &&

# Need to install C++, and export LD_LIBRARY_PATH (~/miniconda3/envs/vl/lib/)
sudo apt install build-essential # installs C++
