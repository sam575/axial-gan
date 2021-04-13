set -ex
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0

default="--dataset_mode aligned_arl3"

for i in {6..7}
do
	# python val.py --model pix2pix --name 001_pix2pix_arl3_polar_nopose_S$i --split $i
	# python val.py --name 001_sagan_arl3_polar_nopose_S$i --split $i
	# python val.py --name 001_axial_arl3_polar_nopose_S$i --split $i
	python val.py --name 001_axial128v2_arl3_polar_nopose_S$i --split $i $default
	sleep 900
done