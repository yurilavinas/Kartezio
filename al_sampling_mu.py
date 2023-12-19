from kartezio.apps.instance_segmentation_mu import create_instance_segmentation_model_mu
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
import random

# active learning, uncertanty metrics
def  count_different_pixels_weighted(array1, array2):
    different_pixels = 0

    for i in range(len(array1)):
        for j in range(len(array1[i])): 
            if array1[i][j] != array2[i][j]:
                if array1[i][j] == 0 or array2[i][j] == 0:
                    different_pixels += 2
                else:
                    different_pixels += 1

    return different_pixels/len(array1)/len(array1[0])

def  count_different_pixels(array1, array2):
    different_pixels = 0

    for i in range(len(array1)):
        for j in range(len(array1[i])): 
            if array1[i][j] != array2[i][j]:
                different_pixels += 1

    return different_pixels/len(array1)/len(array1[0])
# active learning - end

if __name__ == "__main__":
    
    # load data from yml file
    if len(sys.argv) < 2:
        print("Usage\n: python train_model_ative_learning_interactive.py (config, yml file) config.yml (run, int) run")
        sys.exit()
    else:       
        with open(sys.argv[1], "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            framework = cfg["framework"]
            config = cfg["variables"]

    DATASET = framework["DATASET"]  
    RESULTS = framework["save_results"]+"_mu"
    generations = config["generations"]
    CHANNELS = [1, 2]
    preprocessing = SelectChannels(CHANNELS)
    run = sys.argv[2] 
    n_models = config["n_models"]
    _lambda = config["_lambda"]
    _mu = config["_lambda"]
    frequency = config["frequency"]
    indices = config["indices"]
    method = config["method"]
    file_ensemble = f"{RESULTS}/raw_test_data.txt"
    maxeval = config["maxeval"]
    eval_cost = 0
    train_best_ever = 1
    # load yml - end
    
   # mkdir for log data
    try:
        os.makedirs(RESULTS)
        
        data = ["run", "gen", "eval_cost", "test", "train", "best_fitness", "n_active", "used_imgs"]
        with open(file_ensemble, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
    except:
        print()
    # mkdir - done
    
    # getting info: test data and information from the dataset
    indices = np.arange(0, 89).tolist()
    random.shuffle(indices)
    dataset = read_dataset(DATASET, indices=None)
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv
    if preprocessing != None:
        test_x = preprocessing.call(test_x)
    size = len(train_x)

    gen = 0
    # getting info - end
    
    # AL
    idx = [indices.pop()]
    # AL - end
    
   # evolution - start
    while eval_cost <= maxeval:
        print("idx")
        if gen == 0: 
                    
            # init models
            models = [None]*n_models
            gens = [None]*n_models
            for i in range(n_models):    
                models[i] = create_instance_segmentation_model_mu(
                    generations, _mu, _lambda, inputs=2, outputs=2,
                )
                models[i].clear()
                verbose = CallbackVerbose(frequency=frequency)
                callbacks = [verbose]
                if callbacks:
                    for callback in callbacks:
                        callback.set_parser(models[i].parser)
                        models[i].attach(callback)  
            # init - end                 
            
            # init ensemble vars - done here because if we do restart we want the previous info
            strategies = [None]*n_models
            fitness = [None]*n_models
            test_fits = [None]*n_models
            elites = [None]*n_models
            count = 0
            # ensemble - end
    
                
        # load img dataset
        dataset = read_dataset(DATASET, indices=idx)
        train_x, train_y = dataset.train_xy
        if preprocessing != None:
            train_x = preprocessing.call(train_x)
        # load - end

        # evolution
        for i in range(n_models):
            strategies[i], elites[i], gens[i] = models[i].fit(train_x, train_y, elite = elites[i])
            y_hats, _ = models[i].predict(train_x)
            fitness[i] = strategies[i].fitness.compute_one(train_y, y_hats)
        # evolution - end
        
        # gathering test values, for analysis
        for i in range(n_models):
            y_hats, _ = models[i].predict(test_x)
            test_fits[i]  = strategies[i].fitness.compute_one(test_y, y_hats)
        
            if fitness[i] <= train_best_ever:
                train_best_ever = fitness[i]
                test_best_ever = test_fits[i]
                _id = i
        # gathering - end
        
        # saving information for future analysis
        eval_cost += n_models * len(idx) * (len(strategies[0].population.individuals))*gens[i]
            
        active_nodes = models[0].parser.parse_to_graphs(elites[_id])
        data = [run, (gen+1), eval_cost, np.min(test_fits), np.min(fitness), test_best_ever, len(active_nodes[0]+active_nodes[1]), idx]
        with open(file_ensemble, 'a') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(data)
        # saving information - end
                
        
        if indices: # if indices is not empty    
            # active learning methods
            if method == "uncertainty_weighted":
                uncertainties = []
                for img in indices:
                    # loading data
                    dataset = read_dataset(DATASET, indices=[img])
                    x, y = dataset.train_xy
                    # getting masks
                    masks = [None]*n_models
                    for i in range(n_models):
                        masks[i], _ = models[i].predict(x)
                    # masks - end
                    
                    val = 0
                    for i in range(n_models):
                        for j in range(i + 1, n_models):
                            val += count_different_pixels_weighted(masks[i][0]["mask"], masks[j][0]["mask"])
                    uncertainties.append(val) 
                id_ = uncertainties.index(max(uncertainties))    
                idx.append(indices[id_])  
            elif method == "uncertainty":
                uncertainties = []
                for img in indices:
                    # loading data
                    dataset = read_dataset(DATASET, indices=[img])
                    x, y = dataset.train_xy
                    # getting masks
                    masks = [None]*n_models
                    for i in range(n_models):
                        masks[i], _ = models[i].predict(x)
                    # masks - end
                    val = 0
                    for i in range(n_models):
                        for j in range(i + 1, n_models):
                            val += count_different_pixels(masks[i][0]["mask"], masks[j][0]["mask"])
                    uncertainties.append(val) 
                id_ = uncertainties.index(max(uncertainties)) 
                idx.append(indices[id_])   
            elif method == "random":
                if count < size:
                    idx = [count]
                else:
                    if count == size:
                        idx = [np.random.randint(0, size)]
                    rnd = np.random.randint(0, size)
                    idx.append(rnd)
                    # if len(idx) > 10:
                    #     idx.pop(np.random.choice(len(idx),1)[0])
            # AL - end
        
        
        gen += 1
        count += 1
        # evolution - end
            
    # for analysis of the final model
    idx_ = np.argmin(fitness)   
    elite_name = f"{RESULTS}/final_elite_run_{run}_gen_{gen}_model_{idx_}.json"
    models[idx_].save_elite(elite_name, dataset) 
    

    y_hat, _ = models[i].predict(test_x)
    imgs_name = f"{RESULTS}/final_model_run_{run}_gen_{gen}_model_{idx_}.png"
    save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
    
    viewer = KartezioViewer(
        models[idx_].parser.shape, models[i].parser.function_bundle, models[idx_].parser.endpoint
    )
    model_graph = viewer.get_graph(
        elites[idx_], inputs=["In_1","In_2"], outputs=["out_1","out_2"]
    )
    path = f"{RESULTS}/final_graph_model_run_{run}_gen_{gen}_model_{idx_}.png"
    model_graph.draw(path=path)
    # for analysis - end