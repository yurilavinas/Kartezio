
python ../cellpose_mu.py ../configs/config_cgp_gen_100_nmodels_15.yml $1
python ../cellpose_oneplus.py ../configs/config_cgp_gen_100_nmodels_15.yml $1

python ../cellpose_mu.py ../configs/config_cgp_gen_10_nmodels_15.yml $1
python ../cellpose_oneplus.py ../configs/config_cgp_gen_10_nmodels_15.yml $1
