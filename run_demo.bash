#bash

cd face3d/mesh/cython
python3.5 setup.py build_ext -i 

cd ../../../
pwd

python3.5 run_webcam.py
