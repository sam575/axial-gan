set -ex

default="--dataset_mode aligned_arl"

python val.py $default --dir_A ../../dataset/Odin3/8_rec --name 001_pix2pix_arl8 --model pix2pix
sleep 600
python val.py $default --dir_A ../../dataset/Odin3/24_rec --name 001_pix2pix_arl24 --model pix2pix

# python val.py $default --dir_A ../../dataset/Odin3/8_rec --name 001_sagan_arl8 --model pix2pix_sagan
# sleep 600
# python val.py $default --dir_A ../../dataset/Odin3/24_rec --name 001_sagan_arl24 --model pix2pix_sagan

# python val.py $default --dir_A ../../dataset/Odin3/8_rec --name 001_axial_arl8 --model pix2pix_conf
# sleep 600
# python val.py $default --dir_A ../../dataset/Odin3/24_rec --name 001_axial_arl24 --model pix2pix_conf