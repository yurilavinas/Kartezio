import pkg_resources
pkg_resources.require("Kartezio==1.0.0a1")

from kartezio.dataset import read_dataset
from kartezio.plot import save_prediction
from kartezio.activeLearning import active_learning
from kartezio.utils.viewer import KartezioViewer

import sys
import os
import yaml

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage\n: python train_model_ative_learning_interactive.py (config, yml file) config.yml (run, int) run")
        sys.exit()
    else:       
        with open(sys.argv[1], "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        framework = cfg["framework"]
        config = cfg["variables"]

    # framework
    DATASET = framework["DATASET"]
    save_results = framework["save_results"]
    filename = framework["filename"]
    meta_filename = framework["meta_filename"]
    

    generations = config["generations"]

    try:
        os.makedirs(save_results)
    except:
        print('folder already exists, continuing...')

    al = active_learning(cfg)
    al.run = int(sys.argv[2])
    
    for cycle in range(al.iter):
        print("cycle #",cycle,"of AL. Running",al.n_models,"cgp models.") 
        if al.verbose:
            print("using data from", al.lvls)

        # loading data for training
        if al.DATASET == "/tmpdir/lavinas/cellpose":
            CHANNELS = [1, 2]
            dataset = read_dataset(DATASET, indices=al.lvls)
        elif al.DATASET == "/tmpdir/lavinas/ssi":
            dataset = read_dataset(DATASET, indices=al.lvls, filename=filename, meta_filename=meta_filename, preview=False)
            
        # create the output folder for each model - saving elite and population history
        if cycle == 0:
            al.init_model()
        
        train_x, train_y = dataset.train_xy
        if al.DATASET == "/tmpdir/lavinas/cellpose":
            train_x = al.preprocessing.call(train_x)
            train_y = al.preprocessing.call(train_y)
        
        # train with the images available so far - small imgs 
        # either the img is from an annotated area
        # or it's randomly selected - right now i picked the most interesting imgs            
        for i, model in enumerate(al.models):
            file = save_results + "_/internal_model_"+str(i)+"_data.txt"
            al.strategies[i], al.elites[i] = model.fit(train_x, train_y, elite = al.elites[i])

        if al.method == "disagreement":
            al.disagreement = al.calc_disagreement(al.indices) 
        if al.method == "inv":
            al.disagreement = al.calc_disagreement(al.indices) 
        # get the elite from each model
        for i, strategy in enumerate(al.strategies):
            _, al.fitness[i] =  strategy.population.get_best_individual()

        ### saving test data and performance ###
        file = save_results + "/raw_test_data.txt"
        al.saving_test(file, cycle)
        
        # active learning
        # rm img if more than 1 img is in use
        if len(al.lvls) > 1:
            if al.img_t > .0:
                al.rm_img()

        # active learning
        # add img if there are imgs to add
        if len(al.indices) > 0:  # if there are still imgs to add 
            al.add_img()
            
        else:
            al.ask_user = True
            if al.verbose:
                print("no more data to add...")
        
        if al.ask_user: # if there's no data to add, ask the user for more data/evaluation of the model   
            if al.verbose:
                print("asking the user for support...")
            
            # get the mask that the user thought it's the best
            # interactive part
            al.user_input()

            # restart the training ?
            # al.lvls.append(al.idx)
        al.ask_user = False
        
        # update status of the running models
        al.update_stats()
        
        
        al.status = ['continue']*al.n_models


    if al.DATASET == "/tmpdir/lavinas/cellpose":
        dataset_ssi = read_dataset(DATASET, indices=None)
    elif al.DATASET == "/tmpdir/lavinas/ssi":
        dataset = read_dataset(DATASET, indices=al.lvls, filename=filename, meta_filename=meta_filename, preview=False)
        
    test_x, test_y, test_v = dataset.test_xyv
    test_x = al.preprocessing.call(test_x)

    # saving the masks generated by the elites
    # so that we can look and evaluate
    for i, model in enumerate(al.models):  
        model.save_elite(save_results + "/elite_"+str(i)+"_run_"+str(al.run)+".json", dataset_ssi)  
        y_hat, _ = model.predict(test_x)
        imgs_name = save_results+"/gen_"+str(cycle)+"_model_"+str(i)+ "_run_" + str(al.run) + "_.png"
        save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])

        viewer = KartezioViewer(
            model.parser.shape, model.parser.function_bundle, model.parser.endpoint
        )
        model_graph = viewer.get_graph(
            al.elites[i], inputs=["In_1","In_2","In_3"], outputs=["labels"]
        )
        path = save_results+"/graph"+str(cycle)+"_model_"+str(i)+ "_run_" + str(al.run) + "_.png"
        model_graph.draw(path=path)


