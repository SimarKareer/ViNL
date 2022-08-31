conda create -n vl -y python=3.8 &&
conda install -n vl pytorch cudatoolkit=11.6 gdown -c pytorch -c conda-forge -y &&
gdown --fuzzy https://drive.google.com/file/d/1A4DwFb8Jj-PCFcKZE_SJA4b_oN5t9XSm/view?usp=sharing &&
tar -xvf IsaacGym_Preview_3_Package.tar.gz &&
rm IsaacGym_Preview_3_Package.tar.gz &&
pip install -e isaacgym/python &&
cd legged_gym/ &&
pip install -e . &&
pip install tensorboard pytorch_memlab
