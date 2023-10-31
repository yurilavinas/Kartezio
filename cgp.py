import pkg_resources
pkg_resources.require("Kartezio==1.0.0a1")

from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.endpoint import EndpointThreshold
from kartezio.dataset import read_dataset
from kartezio.training import train_model
from kartezio.preprocessing import SelectChannels
import sys
from kartezio.plot import save_prediction
from kartezio.utils.viewer import KartezioViewer
import csv
import os
from kartezio.callback import CallbackVerbose
import yaml
import numpy as np


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage\n: python train_model_ative_learning_interactive.py (config, yml file) config.yml (run, int) run")
        sys.exit()
    else:       
        with open(sys.argv[1], "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            framework = cfg["framework"]
            config = cfg["variables"]
            

    DATASET = framework["DATASET"]  
    RESULTS = framework["save_results"]
    
    cycles = config["iter"]
    generations = config["generations"]

   
    CHANNELS = [1, 2]
    preprocessing = SelectChannels(CHANNELS)
    run = sys.argv[2] 

    n_models = n_models = config["n_models"]
    elites = [None]*n_models
    strategies = [None]*n_models
    fitness = [None]*n_models
    models = [None]*n_models
    file = None
    _lambda = config["_lambda"]
    frequency = config["frequency"]
    indices = config["indices"]
    
    if indices == "None":
        indices = np.random.choice(89, 10).tolist()
    print(indices)

    
    for i in range(n_models):
        models[i] = create_instance_segmentation_model(
            generations, _lambda, inputs=2, outputs=2,
        )
        models[i].clear()
        verbose = CallbackVerbose(frequency=frequency)
        callbacks = [verbose]
        if callbacks:
            for callback in callbacks:
                callback.set_parser(models[i].parser)
                models[i].attach(callback)
    
    
    file_ensemble = f"{RESULTS}/raw_test_data.txt"
    try:
        os.makedirs(RESULTS)
        test = str([f"test_{i}" for i in range(n_models)],).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        train = str([f"train_{i}" for i in range(n_models)]).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        data = ["run", "gen", "eval_cost", "best_test", *test.split(","), *train.split(","),"n_active"]
        with open(file_ensemble, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
    except:
        print('')

    

    dataset = read_dataset(DATASET, indices=indices)
    
    
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv

    train_x = preprocessing.call(train_x)
    test_x = preprocessing.call(test_x)

    for gen in range(cycles):
        for i in range(n_models):
            strategies[i], elites[i] = models[i].fit(train_x, train_y, elite = elites[i])
            y_hats, _ = models[i].predict(train_x)
            fitness[i] = strategies[i].fitness.compute_one(train_y, y_hats)
            
        idx = fitness.index(min(fitness))
        
        test_fits=[None]*n_models
        for i, model in enumerate(models):  
            y_hats, _ = model.predict(test_x)
            test_fits[i]  = strategies[i].fitness.compute_one(test_y, y_hats)

        eval_cost = n_models * (gen+1) * len(train_x) * (len(strategies[i].population.individuals))
        active_nodes = models[0].parser.parse_to_graphs(elites[0])
        data = [run, (gen+1), eval_cost, test_fits[idx], *test_fits, *fitness, len(active_nodes[0]+active_nodes[1])]
        with open(file_ensemble, 'a') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
            

        for i in range(n_models):        

            viewer = KartezioViewer(
                models[i].parser.shape, models[i].parser.function_bundle, models[i].parser.endpoint
            )
            model_graph = viewer.get_graph(
                elites[i], inputs=["In_1","In_2"], outputs=["out_1","out_2"]
            )
            # path = MODELS+"/graph_model_run_" + str(run) + "_.png"
            path = f"{RESULTS}/graph_model_{i}_run_{run}_gen_{gen}.png"
            model_graph.draw(path=path)
    
    for i in range(n_models):        
        elite_name = f"{RESULTS}/final_elite_{i}_gen_{gen}.json"
        models[i].save_elite(elite_name, dataset)  
        y_hat, _ = models[i].predict(test_x)
        imgs_name = f"{RESULTS}/final_model_{i}_run_{run}.png"
        save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
