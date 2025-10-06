for i in $(seq 1 2);
do
    python dataSampling_1plus.py config/test_cellpose.yaml $i
done
