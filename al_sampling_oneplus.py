from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.preprocessing import SelectChannels
from kartezio.plot import save_prediction
import sys
# from kartezio.utils.viewer import KartezioViewer
import csv
import os
from kartezio.callback import CallbackVerbose
import yaml
import numpy as np
import random
import cv2
from scipy.stats import entropy
from itertools import combinations
from PIL import Image
from kartezio.utils import io


def saveElite(model, test_x, run, gen, dataset):
        y_hat, _, _  = model.predict(test_x)
        imgs_name = f"{RESULTS}/elite_image_run_{run}_gen_{gen}_model.png"
        save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
        
        # viewer = KartezioViewer(
        #     model.parser.shape, model.parser.function_bundle, model.parser.endpoint
        # )
        # model_graph = viewer.get_graph(
        #     elite, inputs=["In_1","In_2"], outputs=["out_1","out_2"]
        # )
        # path = f"{RESULTS}/elite_graph_run_{run}_gen_{gen}_model.png"
        # model_graph.draw(path=path)

        name = f"{RESULTS}/elite_run_{run}_gen_{gen}.json"
        model.save_elite(name, dataset) 

def getSolutionsValues(candidates, model, train_x, train_y):
    values = []
    for c in candidates:
        y_hats, _, _ = model.parser.parse(c['model'], train_x)
        c['fitness'] = model.strategy.fitness.compute_one(train_y, y_hats)
        values.append([c['sharpness'],c['fitness']])
    return values

def saveNonDom(candidates, run, gen, train_x, train_y, test_v, model, dataset, file_nondoms):
        
        values = getSolutionsValues(candidates, model, train_x, train_y)
        non_dominated = find_non_dominated_solutions(values)

        data = [run, gen,non_dominated]
        with open(file_nondoms, 'a') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(data)
        
        for i in non_dominated:
            
            y_hat, _, _ = model.parser.parse(candidates[i]['model'], test_x)
            imgs_name = f"{RESULTS}/pf_image_run_{run}_gen_{gen}_model_{i}.png"
            save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
            
            # viewer = KartezioViewer(
            #     model.parser.shape, model.parser.function_bundle, model.parser.endpoint
            # )
            # model_graph = viewer.get_graph(
            #     candidates[i]['model'], inputs=["In_1","In_2"], outputs=["out_1","out_2"]
            # )
            # path = f"{RESULTS}/pf_graph_run_{run}_gen_{gen}_model_{i}.png"
            # model_graph.draw(path=path)
        
            name = f"{RESULTS}/pf_image_run_{run}_gen_{gen}_model_{i}.json"
            model.save_elite(name, dataset) 
            
def find_non_dominated_solutions(solutions):
    non_dominated_solutions = []
    for i, s1 in enumerate(solutions):
        is_dominated = False
        for j, s2 in enumerate(solutions):
            if i == j:
                continue
            # Check if s2 dominates s1
            dominates = True
            for k in range(len(s1)): # Assuming s1 and s2 are lists/tuples of objective values
                if s2[k] > s1[k]:  # Assuming minimization problem (lower is better)
                    dominates = False
                    break
                if s2[k] < s1[k]: # s2 is strictly better in at least one objective
                    pass
            if dominates and any(s2[k] < s1[k] for k in range(len(s1))):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated_solutions.append(i)
    return non_dominated_solutions

def calcUncertainties(method, DATASET, model, future_models, diverseIdx, preprocessing):
    if method == "uncertainty_weighted":
        uncertainties = np.zeros(len(diverseIdx))
        for i, img in enumerate(diverseIdx):
            # Load dataset once per image
            dataset = read_dataset(DATASET, indices=[diverseIdx[i]])
            x, _ = dataset.train_xy
            if preprocessing != None:
                x = preprocessing.call(x)
            # Precompute all masks for this image
            masks = []
            for fm in future_models:
                mask, _, _ = model.parser.parse(fm, x)
                masks.append(mask[0]["mask"])
            # Compute disagreement using itertools.combinations (no nested loops)
            val = sum(
                count_different_pixels_weighted(m1, m2)
                for m1, m2 in combinations(masks, 2)
            )
            uncertainties[i] = val

    elif method == "random":
        uncertainties = np.zeros(len(indices))

    return uncertainties

def getIDx(idx, indices, uncertainties, diverseIdx):
    # if not indices:
    #     return idx, indices

    # Select the index with the highest uncertainty
    id_ = int(np.argmax(uncertainties))
    # indices = indices.tolist()
    idx.append(diverseIdx[id_])
    indices.pop(idx[-1])
    return idx, indices

def find_non_dominated_solutions(solutions):
    non_dominated_solutions = []
    for i, s1 in enumerate(solutions):
        is_dominated = False
        for j, s2 in enumerate(solutions):
            if i == j:
                continue
            # Check if s2 dominates s1
            dominates = True
            for k in range(len(s1)): # Assuming s1 and s2 are lists/tuples of objective values
                if s2[k] > s1[k]:  # Assuming minimization problem (lower is better)
                    dominates = False
                    break
                if s2[k] < s1[k]: # s2 is strictly better in at least one objective
                    pass
            if dominates and any(s2[k] < s1[k] for k in range(len(s1))):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated_solutions.append(i)
    return non_dominated_solutions

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

def count_different_pixels_weighted(array1, array2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)

    # Boolean mask of where they differ
    diff_mask = array1 != array2

    # Case 1: differences where at least one is 0 → weight 2
    weighted_twos = np.sum(((array1 == 0) | (array2 == 0)) & diff_mask) * 2

    # Case 2: differences where neither is 0 → weight 1
    weighted_ones = np.sum((array1 != 0) & (array2 != 0) & diff_mask)

    total_weighted = weighted_twos + weighted_ones

    return total_weighted / array1.shape[0] / array1.shape[1]

def count_different_pixels(array1, array2):
    different_pixels = 0

    for i in range(len(array1)):
        for j in range(len(array1[i])): 
            if array1[i][j] != array2[i][j]:
                different_pixels += 1

    return different_pixels/len(array1)/len(array1[0])

def pearsonr_2D(x, y):
    """computes pearson correlation coefficient"""

    t1= (x - np.mean(x))
    t2= (y - np.mean(y))
    upper = np.sum(t1 * t2)
    lower = np.sqrt(np.sum(np.power(t1,2)) * np.sum(np.power(t2,2)))
    try:
        rho = upper / lower
    except:
        rho=0
    return rho

def sharpness_out(n_noise, y_hats, train_y):
    s_val = np.zeros(n_noise)
    for i in range(n_noise):
        corrs = np.zeros(len(y_hats))
        for j in range(len(y_hats)):
            noise = np.zeros(train_y[0][0].shape,dtype = train_y[0][0].dtype)
            cv2.randn(noise, 0, i+1)
            corrs[j] = np.mean(np.power(pearsonr_2D(y_hats[j]['mask'], train_y[j][0]),2)- np.power(pearsonr_2D(y_hats[j]['mask']+noise, train_y[j][0]),2))
        s_val[i] = np.mean(corrs)
    return np.mean(s_val)

def getNewElite(future_models, model, train_x, train_y):
    fits = np.zeros(len(future_models))
    for i, fm in enumerate(future_models):
        y_hats, _, _ = model.parser.parse(fm, train_x)
        fits[i] = model.strategy.fitness.compute_one(train_y, y_hats)

    fitness = fits[np.argmin(fits)]
    elite = future_models[np.argmin(fits)]
    return elite, fitness

def mutants(elite, n_future, strategy):
    future_models = [None]*n_future
    for i in range(n_future):
        future_models[i] = elite.clone()
        future_models[i] = strategy.mutation_method.mutate(future_models[i])
    future_models.append(elite)
    return future_models

def eval_cost(method, idx, _lambda, n_future, gens, n_diverse):
    if method == "uncertainty_weighted":
        if len(idx)<10:
            eval = (len(idx) - 1)*gens*(_lambda) + (n_future+1)*n_diverse  + (n_future+1)*len(idx)
        else:
            eval = (len(idx) - 1)*gens*(_lambda)  
    elif method == "random":
        eval = (len(idx) - 1) * gens * (_lambda)
    return eval
        
def getFit(model, x, y):
    y_hats, _, _ = model.predict(x)
    return strategy.fitness.compute_one(y, y_hats)

def diverseImage(featureSpace, selectedFeatureSpace, numImages=1):
    min_distances = [np.min([np.linalg.norm(f1 - f2) for f1 in selectedFeatureSpace]) for f2 in featureSpace]
    most_diverse_index = np.argsort(min_distances)
    most_diverse_index = most_diverse_index[::-1]
    return most_diverse_index[:numImages]

def diverseImagesIterative(featureSpace, selectedFeatureSpace, numImages=2):
    diverseIndicesList = []
    for i in range(0, numImages):
        diverseIndex = diverseImage(featureSpace, selectedFeatureSpace, 1)
        # diverseImagesList.append(diverseImages[0])
        diverseIndicesList.append(diverseIndex[0].item())
        selectedFeatureSpace = np.vstack([selectedFeatureSpace, featureSpace[diverseIndex[0]]])
    return diverseIndicesList

def typicalPoint(featureSpace, k=10):
    densities = []
    for i in range(len(featureSpace)):
        distances = []
        for j  in range(len(featureSpace)):
            if i!=j:
                distances.append(np.linalg.norm(featureSpace[i] - featureSpace[j]))
        densities.append(np.mean(sorted(distances)[:k]))
    typical_index = np.argmin(densities)
    return typical_index


if __name__ == "__main__":
    
    # load data from yml file
    if len(sys.argv) < 2:
        print("Use\n: python train_model_ative_learning_interactive.py (config, yml file) config.yml (run, int) run")
        sys.exit()
    else:       
        with open(sys.argv[1], "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            framework = cfg["framework"]
            config = cfg["variables"]
            
    DATASET = framework["DATASET"]  
    RESULTS = framework["save_results"]+"_oneplus"
    generations = config["generations"]
    CHANNELS = [1, 2]
    preprocessing = SelectChannels(CHANNELS)
    run = sys.argv[2] 

    _lambda = config["_lambda"]
    n_mutations = config["n_mutations"]
    frequency = config["frequency"]
    method = config["method"]
    file_raw_data = f"{RESULTS}/raw_test_data.txt"
    file_nondoms = f"{RESULTS}/nondoms_{run}/nondoms.txt"
    maxeval = config["maxeval"]
    n_future = config["n_future"]
    n_noise = config["n_noise"]
    n_diverse = config["n_diverse"]
    init_idx = config["init_idx"]
    img_limit = config["img_limit"]
    # mkdir for log data
    try:
        os.makedirs(RESULTS)
        
        data = ["init_idx, run, gen, eval, lambda, train, test, size, idx, uncertainty, sharpness, updatedElite"]
        with open(file_raw_data, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)


        data = ["run, gen, non_dominated"]
        with open(file_nondoms, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
    except:
        print()
    # mkdir - done


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
    

    # getting info: test data and information from the dataset
    indices = np.arange(0, 89).tolist()
    

    # pixels = np.loadtxt(f"/Users/yurilavinas/Documents/MCF/datasets/cellpose/features.txt")
    pixels = np.loadtxt(f"/tmpdir/lavinas/datasets/cellpose/features.txt")
    
    if init_idx == 'typical':
        init_idx = typicalPoint(pixels, k=10)
        idx = [indices.pop(init_idx)]    
    elif init_idx == "cluster":
        from sklearn.cluster import KMeans
        import pandas as pd
        labels = [f'{i}' for i in range(89)]
        df_ = pd.DataFrame(pixels)
        df_['Label'] = labels # Annotate each point
        kmeans = KMeans(n_clusters=6).fit(pixels)
        df_['cluster'] = pd.Categorical(kmeans.labels_)
        tmp=[int(np.random.choice(np.asarray(df_.iloc[kmeans.labels_==l,:]['Label']),1)[0]) for l in np.unique(kmeans.labels_)]
        tmp.sort(reverse = True)
        idx=[]
        for id_ in tmp: 
            idx.append(indices.pop(id_))
    elif init_idx == 'rnd':
        random.shuffle(indices)
        idx = [indices.pop()]
    else:
        idx=None
    dataset = read_dataset(DATASET, indices=idx)
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv
    if preprocessing != None:
        train_x = preprocessing.call(train_x)
        test_x = preprocessing.call(test_x)

    candidates = []
    elite = None
    updatedElite = 0 
    gen = 0
    eval = 0
    uncertainties = 0
    sharpness = 0
    print(idx)
    while eval <= maxeval:
        print("==================")
        print("generation: ",gen+1)
        print("------------------")
                    
        strategy, gens = model.fit(train_x, train_y, elite = elite, gen = generations)
        elite = strategy.elite
        fitness = np.min(strategy.population.fitness)
        #cost: len(idx)*gen*_lambda
        test_fits = getFit(model, test_x, test_y)
        
        if len(idx) < img_limit:
            future_models = mutants(elite, n_future, strategy)
            #cost: 0

            diverseIdx = diverseImagesIterative(pixels, pixels[idx], n_diverse)
            
            #cost: 0
            uncertainties = calcUncertainties(method, DATASET, model, future_models, diverseIdx, preprocessing)
            #cost: future_models*len(diverseIdx) 
        
            idx, indices = getIDx(idx, indices, uncertainties, diverseIdx)
            #cost: 0
            dataset = read_dataset(DATASET, indices=idx)
            train_x, train_y = dataset.train_xy
            if preprocessing != None:
                train_x = preprocessing.call(train_x)
            
            newElite, fitness = getNewElite(future_models, model, train_x, train_y)
            if newElite!=elite:
                updatedElite += 1
                model.strategy.population.set_elite(elite)
            #cost: future_models*len(idx)
    
        y_hats, _ , _ = model.predict(train_x)
        sharpness = sharpness_out(n_noise,y_hats, train_y)
        
        eval += eval_cost(method, idx, _lambda, n_future, gens, n_diverse)
        
        solution = {'model':elite,'sharpness':sharpness,'fitness':fitness, 'test_fitness': test_fits}
        candidates.append(solution)
        
        active_nodes = model.parser.parse_to_graphs(elite)
        gen += 1
        data = [init_idx, run, gen, eval, _lambda, fitness, test_fits, len(active_nodes[0]+active_nodes[1]), idx, np.max(uncertainties), sharpness, updatedElite]
        with open(file_raw_data, 'a') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(data)

        

    print("saving non dominated...")
    saveNonDom(candidates, run, gen, train_x, train_y, test_v, model, dataset, file_nondoms)
    print("saving elite...")
    saveElite(model, test_x, run, gen, dataset)

    