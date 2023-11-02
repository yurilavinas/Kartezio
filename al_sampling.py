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

    n_models = config["n_models"]
    _lambda = config["_lambda"]
    frequency = config["frequency"]
    indices = config["indices"]
    method = config["method"]
    file_ensemble = f"{RESULTS}/raw_test_data.txt"
    c = config["c"]
    a = config["a"] 
    thres = config["t"]
    restart = config["restart"]
    val = config["val"]
    eval_cost = 0
    train_best_ever = 1
    
    try:
        os.makedirs(RESULTS)
        
        test = str([f"test_{i}" for i in range(n_models)],).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        train = str([f"train_{i}" for i in range(n_models)]).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        data = ["run", "gen", "eval_cost", *test.split(","), *train.split(","), "best_fitness", "n_active", "idx"]
        with open(file_ensemble, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
    except:
        print()

    
    indices = np.arange(0, 89).tolist()
    dataset = read_dataset(DATASET, indices=None)
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv
    if preprocessing != None:
        test_x = preprocessing.call(test_x)
    size = len(train_x)
    
    
    for gen in range(cycles):
        if gen == 0 or (restart == True and eval_cost > val): 
            if gen > 0:
                viewer = KartezioViewer(
                    model.parser.shape, model.parser.function_bundle, model.parser.endpoint
                )
                model_graph = viewer.get_graph(
                    elites, inputs=["In_1","In_2"], outputs=["out_1","out_2"]
                )
                path = f"{RESULTS}/graph_model_run_{run}_gen_{gen}.png"
                model_graph.draw(path=path)
                
            # print("init!!!")  
            model = create_instance_segmentation_model(
                generations, _lambda, inputs=2, outputs=2,
            )
            model.clear()
            verbose = CallbackVerbose(frequency=frequency)
            callbacks = [verbose]
            if callbacks:
                for callback in callbacks:
                    callback.set_parser(model.parser)
                    model.attach(callback)
                    
            
            probs = np.ones(size)
            probs_uniq = np.ones(size)
            probs_inv = np.ones(size)
            count = 0
            elites = None
        
        if count < size:
            idx = [count]
            count += 1
        else:
            
            if method == "ranking":
                tmp1 = [np.random.choice(np.flatnonzero(probs == probs.max())).tolist()]
                tmp2 = [np.random.choice(np.flatnonzero(probs == probs.min())).tolist()]
                idx = [tmp1[0], tmp2[0]]
            elif method == "roulette_only_worse":
                tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                idx = [tmp1[0]]
            elif method == "roulette":
                tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                tmp2 = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()
                idx = [tmp1[0], tmp2[0]]
            elif method == "ranking_inc":
                if gen == size:
                    tmp1 = [np.random.choice(np.flatnonzero(probs == probs.max())).tolist()]
                    tmp2 = [np.random.choice(np.flatnonzero(probs == probs.min())).tolist()]
                    idx = [tmp1[0], tmp2[0]]
                f = int((c*gen+1)**a)   
                if len(idx) >= 10:
                    tmp1 = [np.random.choice(np.flatnonzero(probs == probs.max())).tolist()]
                    tmp2 = [np.random.choice(np.flatnonzero(probs == probs.min())).tolist()]
                    idx = [tmp1[0], tmp2[0]]
                elif gen % f == 0: 
                    tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()    
                    idx.append(tmp1[0])
            elif method == "roullet_inc":
                if gen == size:
                    tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                    tmp2 = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()
                    idx = [tmp1[0], tmp2[0]]
                f = int((c*gen+1)**a)   
                for i, cand in enumerate(probs_uniq[idx]):
                    if cand < 0.1:
                        idx.pop(i)
                if gen % f == 0 or len(idx)==0: 
                    tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()  
                    tmp2 = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()  
                    idx.append(tmp1[0])
                    idx.append(tmp2[0])
            elif method == "roulette_inc_del":
                if gen == size:
                    tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                    tmp2 = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()
                    idx = [tmp1[0], tmp2[0]]
                f = int((c  *gen+1)**a)   
                for i, cand in enumerate(probs_uniq[idx]):
                    if cand < thres:
                        idx.pop(i)
                        break
                if gen % f == 0 or len(idx) == 0: 
                    tmp1 = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                    tmp2 = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()
                    idx.append(tmp1[0])
                    idx.append(tmp2[0])
                if len(idx) > 10:
                    idx.pop(np.random.choice(len(idx), 1)[0])
                    idx.pop(np.random.choice(len(idx), 1)[0]) 
                    
        # print(count, idx)
        dataset = read_dataset(DATASET, indices=idx)
        train_x, train_y = dataset.train_xy
        if preprocessing != None:
            train_x = preprocessing.call(train_x)

        strategy, elites = model.fit(train_x, train_y, elite = elites)
        y_hats, _ = model.predict(train_x)
        fitness = strategy.fitness.compute_one(train_y, y_hats)
              
        for _idx in idx:
            __idx = _idx
            probs[__idx] = fitness
            probs_inv[__idx] = fitness
        probs_uniq[idx[0]] = fitness
        
        y_hats, _ = model.predict(test_x)
        test_fits  = strategy.fitness.compute_one(test_y, y_hats)
        
        if fitness <= train_best_ever:
            train_best_ever = fitness
            idx_best_ever = idx[0]
            test_best_ever = test_fits

        eval_cost += n_models * len(idx) * (len(strategy.population.individuals))
        active_nodes = model.parser.parse_to_graphs(elites)
        data = [run, (gen+1), eval_cost, test_fits, fitness, test_best_ever, len(active_nodes[0]+active_nodes[1]), idx]

        with open(file_ensemble, 'a') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
            
        
    
    elite_name = f"{RESULTS}/final_elite.json"
    model.save_elite(elite_name, dataset) 
    

    y_hat, _ = model.predict(test_x)
    imgs_name = f"{RESULTS}/final_model_run_{run}_gen_{gen}_.png"
    save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
    
    viewer = KartezioViewer(
        model.parser.shape, model.parser.function_bundle, model.parser.endpoint
    )
    model_graph = viewer.get_graph(
        elites, inputs=["In_1","In_2"], outputs=["out_1","out_2"]
    )
    path = f"{RESULTS}/final_graph_model_run_{run}_gen_{gen}.png"
    model_graph.draw(path=path)