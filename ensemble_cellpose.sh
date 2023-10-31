# printf "run, cycle, eval_cost, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, best, train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, dis, patch_id" >> /tmpdir/lavinas/60nodes_1000gen_5lambda_10model/raw_test_data.txt
python ensemble_cellpose.py config_cgp_ensemble.yml $1
