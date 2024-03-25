from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
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

def selLexicase(values, images, k, maximizing = True):
    ### heavily based on DEAP's implementation
    ### https://github.com/DEAP/deap/blob/master/deap/tools/selection.py
    
    """
    Returns an individual that does the best on the fitness cases when considered one at a
    time in random order.
    https://push-language.hampshire.edu/uploads/default/original/1X/35c30e47ef6323a0a949402914453f277fb1b5b0.pdf
    Implemented lambda_epsilon_y implementation.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    
    selected_images_id = []
    for _ in range(k):
        candidates = np.arange(0, len(images)).tolist()
        cases = list(range(len(values)))
        random.shuffle(cases)
        

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = values[cases[0][candidates]]
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            
            if maximizing:
                best_val_for_case = max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = [x for x in range(len(candidates)) if values[cases[0]][x] >= min_val_to_survive]
            else:
                best_val_for_case = min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = [x for x in range(len(candidates)) if values[cases[0]][x] <= max_val_to_survive]

            cases.pop(0)
        
        
        if k == 1:
            selected_images_id = random.choice(candidates)
        else:
            selected_images_id.append(random.choice(candidates))
    
    return images[selected_images_id]

# active learning, uncertanty metrics
def count_different_pixels_weighted(array1, array2):
    different_pixels = 0

    for i in range(len(array1)):
        for j in range(len(array1[i])): 
            if array1[i][j] != array2[i][j]:
                if array1[i][j] == 0 or array2[i][j] == 0:
                    different_pixels += 2
                else:
                    different_pixels += 1

    return different_pixels/len(array1)/len(array1[0])

def models_disagreement(array1, array2):
    disagreement = np.sum(array1 != array2) / np.prod(array1.shape)
    return disagreement

def variance_disagreement(array1, array2):
    # Calculate the variance of pixel values for each mask
    variance_array1 = np.var(array1[0])
    variance_array2 = np.var(array2[0])

    # Calculate the absolute difference in variance
    disagreement = np.abs(variance_array1 - variance_array2)

    return disagreement

def models_entropy(array1, array2):
    from scipy.stats import entropy
    # # Flatten the segmentation masks into 1D arrays
    # flat_mask1 = array1.flatten()
    # flat_mask2 = array2.flatten()

    # Compute entropy for each mask
    if (np.sum(array1)) != 0:
        entropy_mask1 = entropy(array1[0], base = 2)
    else:
        entropy_mask1 = 0
        
    if (np.sum(array2)) != 0:
        entropy_mask2 = entropy(array2[0], base = 2)
    else:
        entropy_mask2 = 0
    
    # Calculate the absolute difference in entropy
    disagreement = np.abs(entropy_mask1 - entropy_mask2)

    if np.isnan(disagreement):
        disagreement = 0 

    return disagreement

def count_different_pixels(array1, array2):
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
    RESULTS = framework["save_results"] 
    generations = config["generations"]
    CHANNELS = [1, 2]
    preprocessing = SelectChannels(CHANNELS)
    run = sys.argv[2] 
    n_models = config["n_models"]
    _lambda = config["_lambda"]
    _mu = config["_lambda"]
    frequency = config["frequency"]
    # indices = config["indices"]
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
    indices_all = np.arange(0, 89).tolist()
    random.shuffle(indices)
    random.shuffle(indices_all)
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
    idx_all = []
    # AL - end
    
   # evolution - start
    while eval_cost <= maxeval:
        print("\nTraining")
        print("idx", idx)
        if gen == 0: 
                    
            # init models
            models = [None]*n_models
            gens = [None]*n_models
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
            test_fits[i] = strategies[i].fitness.compute_one(test_y, y_hats)
        
            if fitness[i] <= train_best_ever:
                train_best_ever = fitness[i]
                test_best_ever = test_fits[i]
        # gathering - end
        
        # saving information for future analysis
        eval_cost += n_models * len(idx) * (len(strategies[0].population.individuals))*gens[i]
            
        active_nodes = models[0].parser.parse_to_graphs(elites[np.argmin(fitness)])
        data = [run, (gen+1), eval_cost, np.min(test_fits), np.min(fitness), test_best_ever, len(active_nodes[0]+active_nodes[1]), idx]
        with open(file_ensemble, 'a') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(data)
        # saving information - end
                
          
        # active learning methods
        count = []
        count_w = []
        entropy = []
        disagreement = []
        for img in indices:
            # loading data
            dataset = read_dataset(DATASET, indices=[img])
            x, y = dataset.train_xy
            # getting masks
            masks = [None]*n_models
            for i in range(n_models):
                masks[i], _ = models[i].predict(x)
            # masks - end
            
            val_count = 0
            val_count_w = 0
            val_entropy = 0
            val_disagreement = 0
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    val_count_w += count_different_pixels_weighted(masks[i][0]["mask"], masks[j][0]["mask"])
                    val_entropy += models_entropy(masks[i][0]["mask"], masks[j][0]["mask"])
                    val_disagreement += variance_disagreement(masks[i][0]["mask"], masks[j][0]["mask"])
                    val_count += count_different_pixels(masks[i][0]["mask"], masks[j][0]["mask"])
            count_w.append(val_count_w) 
            count.append(val_count) 
            entropy.append(val_entropy) 
            disagreement.append(val_disagreement) 
        print("--------------------------------------------------------------------------------------------------------------------")
        print("AL")
        # print('count, entropy, variance')
        # print(count, entropy, disagreement)
        # print(indices[count.index(max(count))], indices[entropy.index(max(entropy))], indices[disagreement.index(max(disagreement))])
        # id_count = count.index(max(count))
        # id_entropy = entropy.index(max(entropy))
        # id_disagremment = disagreement.index(max(disagreement))
        # id_ = [id_count, id_entropy, id_disagremment]
        id_ = selLexicase(values=[count, count_w, entropy, disagreement], images=indices, k=1)
        idx.append(id_)
        indices.remove(id_)
        print("--------------------------------------------------------------------------------------------------------------------")
        # id_ = np.unique(id_)
        # id_ = np.sort(id_)[::-1]
        # for i in id_:
        #     idx.append(indices.pop(i))
            # idx.append(indices[i])  # with rep   
        # if len(idx) > 10:
        #     indices = np.arange(0, 89).tolist()
        #     random.shuffle(indices)
        #     idx = [indices.pop()]
        # AL - end
        
        # count = []
        # entropy = []
        # disagreement = []
        # for img in indices_all:
        #     # loading data
        #     dataset = read_dataset(DATASET, indices=[img])
        #     x, y = dataset.train_xy
        #     # getting masks
        #     masks = [None]*n_models
        #     for i in range(n_models):
        #         masks[i], _ = models[i].predict(x)
        #     # masks - end
            
        #     val_count = 0
        #     val_entropy = 0
        #     val_disagreement = 0
        #     for i in range(n_models):
        #         for j in range(i + 1, n_models):
        #             val_count += count_different_pixels_weighted(masks[i][0]["mask"], masks[j][0]["mask"])
        #             val_entropy += models_entropy(masks[i][0]["mask"], masks[j][0]["mask"])
        #             val_disagreement += variance_disagreement(masks[i][0]["mask"], masks[j][0]["mask"])
        #     count.append(val_count) 
        #     entropy.append(val_entropy) 
        #     disagreement.append(val_disagreement) 
        # # print('count, entropy, disagreement')
        # # print(indices[count.index(max(count))], indices[entropy.index(max(entropy))], indices[disagreement.index(max(disagreement))])
        # id_count = count.index(max(count))
        # id_entropy = entropy.index(max(entropy))
        # id_disagremment = disagreement.index(max(disagreement))
        # id_ = [id_count, id_entropy, id_disagremment]
        # # print([indices[item] for item, count in collections.Counter(id_all).items() if count > 1])
        # # print("--------------------------------------------------------------------------------------------------------------------")
        # id_ = np.unique(id_)
        # id_ = np.sort(id_)[::-1]
        # for i in id_:
        #     idx_all.append(indices_all[i])
        # # recommendation - end
        # print("recommending id")
        # print(idx_all)
        # print("--------------------------------------------------------------------------------------------------------------------")
        
        gen += 1
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