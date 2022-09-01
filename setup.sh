mamba create -n vl -y python=3.8 &&
conda activate vl &&
mamba install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y &&
mamba install gdown tensorboard -c conda-forge -y &&
gdown --fuzzy https://drive.google.com/file/d/1A4DwFb8Jj-PCFcKZE_SJA4b_oN5t9XSm/view?usp=sharing &&
tar -xvf IsaacGym_Preview_3_Package.tar.gz &&
rm IsaacGym_Preview_3_Package.tar.gz &&
pip install -e isaacgym/python rsl_rl/. &&
pip install -e legged_gym/. &&
pip install pytorch-memlab
