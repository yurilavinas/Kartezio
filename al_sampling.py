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
    

    generations = config["generations"]

   
    CHANNELS = [1, 2]
    preprocessing = SelectChannels(CHANNELS)
    run = sys.argv[2] 

    n_models = config["n_models"]
    _lambda = int(config["_lambda"]/2)
    frequency = config["frequency"]
    indices = config["indices"]
    method = config["method"]
    file_ensemble = f"{RESULTS}/raw_test_data.txt"
    c = config["c"]
    a = config["a"] 
    thres_hard = config["t_hard"]
    thres_easy = config["t_easy"]
    restart = config["restart"]
    maxeval = config["maxeval"]
    val = config["val"]
    checkpoint = 0
    eval_cost = 0
    train_best_ever = 1
    
    try:
        os.makedirs(RESULTS)
        
        # test = str([f"test_{i}" for i in range(1)],).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        # train = str([f"train_{i}" for i in range(1)]).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        data = ["run", "gen", "eval_cost", "test", "train", "best_fitness", "n_active", "idx"]
        with open(file_ensemble, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
    except:
        print()

    
    indices = np.arange(0, 89).tolist()
    dataset = read_dataset(DATASET, indices=[0])
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv
    if preprocessing != None:
        test_x = preprocessing.call(test_x)
    size = len(train_x)
    
    print(dataset)
    print(train_x)
    print(train_y)
    print(np.unique(train_y))
    exit()
    
    gen = 0
    
    while eval_cost < maxeval:
        if gen == 0 or (restart == True and eval_cost > checkpoint): 
            models = [None]*n_models
            strategies = [None]*n_models
            fitness = [None]*n_models
            test_fits = [None]*n_models
            if gen > 0:
                for i in range(n_models):
                    elite_name = f"{RESULTS}/restart_elite_run_{run}_gen_{gen}.json"
                    models[i].save_elite(elite_name, dataset) 
                    
                    viewer = KartezioViewer(
                        models[i].parser.shape, models[i].parser.function_bundle, models[i].parser.endpoint
                    )
                    model_graph = viewer.get_graph(
                        elites[i], inputs=["In_1","In_2"], outputs=["out_1","out_2"]
                    )
                    path = f"{RESULTS}/restart_graph_model_run_{run}_gen_{gen}.png"
                    model_graph.draw(path=path)
                    
            
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
            
            probs = np.ones(size)
            probs_uniq = np.ones(size)
            probs_inv = np.ones(size)
            
            elites = [None]*n_models
            checkpoint = checkpoint + val
            count = 0
        
        
        
        if count < size:
            idx = [count]
        
        elif method == "random":
            if count == size:
                idx = [np.random.randint(0, size)]
            rnd = np.random.randint(0, size)
            idx.append(rnd)
            if len(idx) > 10:
                idx.pop(np.random.choice(len(idx),1)[0])
        elif method == "roulette_inc_del":
            if count == size:
                well_perf = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()
                bad_perf = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                idx = [well_perf[0], bad_perf[0]]
            elif count > size:
                f = int((c*gen+1)**a)   
                if gen % f == 0 or len(idx) == 0: 
                    well_perf = np.random.choice(size, 1, p=np.array(probs_inv)/sum(probs_inv)).tolist()
                    bad_perf = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                    idx.append(well_perf[0])
                    idx.append(bad_perf[0])
                if len(idx) > 10:
                    idx.pop(np.random.choice(len(idx),1)[0])
                    idx.pop(np.random.choice(len(idx),1)[0])
        
                for i, cand in enumerate(probs_uniq[idx]): # removing images that don't pose a challenge
                    if cand < thres_easy:
                        idx.pop(i) 
                
        dataset = read_dataset(DATASET, indices=idx)
        train_x, train_y = dataset.train_xy
        if preprocessing != None:
            train_x = preprocessing.call(train_x)

        for i in range(n_models):
            strategies[i], elites[i] = models[i].fit(train_x, train_y, elite = elites[i])
            y_hats, _ = models[i].predict(train_x)
            fitness[i] = strategies[i].fitness.compute_one(train_y, y_hats)

        for _idx in idx:
            __idx = _idx
            probs[__idx] = np.min(fitness)
            probs_inv[__idx] = 1 - np.min(fitness)
        probs_uniq[idx[0]] = 1 - np.min(fitness)
        
        for i in range(n_models):
            y_hats, _ = models[i].predict(test_x)
            test_fits[i]  = strategies[i].fitness.compute_one(test_y, y_hats)
        
            if fitness[i] <= train_best_ever:
                train_best_ever = fitness[i]
                idx_best_ever = idx[0]
                test_best_ever = test_fits[i]

        eval_cost += n_models * len(idx) * (len(strategies[i].population.individuals))
        active_nodes = models[i].parser.parse_to_graphs(elites[i])
        data = [run, (gen+1), eval_cost, np.min(test_fits), np.min(fitness), test_best_ever, len(active_nodes[0]+active_nodes[1]), idx]
        with open(file_ensemble, 'a') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(data)
                
        gen += 1
        count += 1
        
        if eval_cost % 1000 == 0:
             
            idx = np.argmin(fitness)
            elite_name = f"{RESULTS}/elite_run_{run}_gen_{gen}_model_{idx}.json"
            models[idx].save_elite(elite_name, dataset) 
            
            y_hat, _ = models[idx].predict(test_x)
            imgs_name = f"{RESULTS}/model_run_{run}_gen_{gen}_model_{idx}.png"
            save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
        
            viewer = KartezioViewer(
                models[idx].parser.shape, models[idx].parser.function_bundle, models[idx].parser.endpoint
            )
            model_graph = viewer.get_graph(
                elites[i], inputs=["In_1","In_2"], outputs=["out_1","out_2"]
            )
            path = f"{RESULTS}/graph_model_run_{run}_gen_{gen}_model_{i}.png"
            model_graph.draw(path=path)
            
    
    idx = np.argmin(fitness)   
    elite_name = f"{RESULTS}/final_elite_run_{run}_gen_{gen}_model_{idx}.json"
    models[idx].save_elite(elite_name, dataset) 
    

    y_hat, _ = models[i].predict(test_x)
    imgs_name = f"{RESULTS}/final_model_run_{run}_gen_{gen}_model_{idx}.png"
    save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
    
    viewer = KartezioViewer(
        models[idx].parser.shape, models[i].parser.function_bundle, models[idx].parser.endpoint
    )
    model_graph = viewer.get_graph(
        elites[idx], inputs=["In_1","In_2"], outputs=["out_1","out_2"]
    )
    path = f"{RESULTS}/final_graph_model_run_{run}_gen_{gen}_model_{idx}.png"
    model_graph.draw(path=path)