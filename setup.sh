mamba create -n vl -y python=3.8 &&
mamba install -n vl pytorch cudatoolkit=11.7 gdown tensorboard pytorch_memlab -c pytorch -c conda-forge -y &&
gdown --fuzzy https://drive.google.com/file/d/1A4DwFb8Jj-PCFcKZE_SJA4b_oN5t9XSm/view?usp=sharing &&
tar -xvf IsaacGym_Preview_3_Package.tar.gz &&
rm IsaacGym_Preview_3_Package.tar.gz &&
conda activate vl &&
pip install -e isaacgym/python &&
pip install -e legged_gym/. &&
pip install -e rsl_rl/.
