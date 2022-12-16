conda create -n vinl -y python=3.7 &&
conda activate vinl &&
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y &&
conda install gdown tensorboard -c conda-forge -y &&
pip install -r requirements.txt &&
cd submodules && 
gdown --fuzzy https://drive.google.com/file/d/1A4DwFb8Jj-PCFcKZE_SJA4b_oN5t9XSm/view?usp=sharing &&
tar -xvf IsaacGym_Preview_3_Package.tar.gz &&
rm IsaacGym_Preview_3_Package.tar.gz &&
cd isaacgym/python && pip install -e . && cd - &&
cd rsl_rl && pip install -e . && cd - &&

# Habitat Lab installs
cd habitat-lab &&
pip install -r requirements.txt &&
python setup.py develop --all &&
git checkout aliengoisaac &&
conda install -c conda-forge bullet -y &&
cd - && cd .. &&

# legged_gym install
pip install -e . &&

# Need to install C++, and export LD_LIBRARY_PATH (~/miniconda3/envs/vinl/lib/)
sudo apt install build-essential # installs C++
