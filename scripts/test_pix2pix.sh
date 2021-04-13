set -ex
#python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch

default="--dataset_mode aligned_arl3"

# name="001_pix2pix_arl3_polar_nopose_S"
# args="--model pix2pix --dir_A ../../dataset/odin_data/Polar_128_rec/"
#
# name="001_sagan_arl3_polar_nopose_S"
# args="--netG SAGAN --dir_A ../../dataset/odin_data/Polar_128_rec/"

# name="001_axial8_arl3_polar_nopose_S"
# args="--netG axial --dir_A ../../dataset/odin_data/Polar_16/"

name="001_axial128v2_arl3_polar_nopose_S"
args="--netG axial_128v2 --dir_A ../../dataset/odin_data/Polar_128/"

for i in {5..7}
do
	python test_arl3.py --name $name$i --split $i $default $args --epoch best
	python test_arl3.py --name $name$i --split $i $default $args --epoch best_ssim
	python test_arl3.py --name $name$i --split $i $default $args --epoch best_id
	# sleep 10
done