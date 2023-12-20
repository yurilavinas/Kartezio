
python ../al_sampling_mu.py ../configs/config_uncertainty_gen_100_nmodels_5.yml $1
python ../al_sampling_oneplus.py ../configs/config_uncertainty_gen_100_nmodels_5.yml $1

python ../al_sampling_mu.py ../configs/config_uncertainty_gen_10_nmodels_5.yml $1
python ../al_sampling_oneplus.py ../configs/config_uncertainty_gen_10_nmodels_5.yml $1
