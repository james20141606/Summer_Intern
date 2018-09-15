

module load cuda/9.0-fasrc01
module load cudnn/7.0.3-fasrc02

module load Anaconda

source ~/anaconda2/bin/activate kears_theano


THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True python -u test_pixelwise.py --imagedir test_data/grayscale_maps_half/ --savename test_data/jwr_pixelwise_polarity.h5 --modelname models/3D_unet_jwr_synapse_polarity_half_linear_leaky_f24_316_32_100000.json --weightname models/3D_unet_jwr_synapse_polarity_half_linear_leaky_f24_316_32_100000_weights.h5



