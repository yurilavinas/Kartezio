from kartezio.apps.segmentation import create_segmentation_model
from kartezio.endpoint import EndpointThreshold
from kartezio.dataset import read_dataset
import sys
from kartezio.plot import save_prediction
import csv
import os
from kartezio.callback import CallbackVerbose
import yaml
import numpy as np
import random
import cv2
from numena.image.color import gray2rgb
import albumentations as A
import pandas as pd

# Declare an augmentation pipeline
transform = A.Compose([
    A.GaussianBlur(p=0.2),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    A.ShiftScaleRotate(p=0.2),
    A.OpticalDistortion(p=0.2),
    A.CLAHE(p=0.2),
    A.Transpose(p=0.2),
    A.CoarseDropout(max_holes=5, max_height=2, max_width=2, fill_value=64, p=0.2)
])

def datasetwrite(i, filename):
    with open(filename, 'a', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        data = [f'{i}_img.png',f'{i}_masks.png','training']
        writer.writerow(data)


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
        print("Use\n: python train_model_ative_learning_interactive.py (config, yml file) config.yml (run, int) run")
        sys.exit()
    else:       
        with open(sys.argv[1], "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            framework = cfg["framework"]
            config = cfg["variables"]
            
    DATASET = framework["DATASET"]  
    RESULTS = framework["save_results"]
    generations = config["generations"]
    # CHANNELS = [1, 2]
    # preprocessing = SelectChannels(CHANNELS)
    run = sys.argv[2] 

    n_models = config["n_models"]
    _lambda = config["_lambda"]
    frequency = config["frequency"]
    method = config["method"]
    file_ensemble = f"{RESULTS}/raw_test_data.txt"
    maxeval = config["maxeval"]
    aug = False
    if aug:
        AUGSET = f"{DATASET}/aug"
        os.system(f"rm -r {AUGSET}")
        os.makedirs(AUGSET)
        cmd = f'cp "{DATASET}/META.json" "{AUGSET}/META.json"'
        os.system(cmd)
        with open(f'{AUGSET}/dataset.csv', 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            header = ["input","label","set"]
            writer.writerow(header)
    num = 0
    eval_cost = 0
    e = 0.05
    # load yml - end
    endpoint = EndpointThreshold(threshold=128)
    
    # mkdir for log data
    try:
        os.makedirs(RESULTS)
        
        data = ["run", "gen", "eval_cost", "test", "train", "n_active", "used_imgs", "bests_ids", "worses_ids", "uncertainty", "image"]
        with open(file_ensemble, 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
    except:
        print()
    # mkdir - done

    # getting info: test data and information from the dataset
    dataset = read_dataset(DATASET, indices=None)
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv
    # if preprocessing != None:
    #     test_x = preprocessing.call(test_x)
    indices = np.arange(0, len(train_x)).tolist()
    
    if config["max_imgs"] =='all':
        max_imgs = len(indices)
    else:
        max_imgs = config["max_imgs"]
        
    old_var = -1
    gen = 0
    # getting info - end
    
    # AL
    idx = [indices.pop(0)]
    random.shuffle(indices)
    # AL - end
    # evolution - start
    while eval_cost <= maxeval:
        print("gen",gen+1, 'with', idx)
        if gen == 0: 
            
            # init models
            models = [None]*n_models
            gens = [None]*n_models
            for i in range(n_models):   
                models[i] = create_segmentation_model(
                    generations, _lambda, inputs=3, outputs=1,
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
            fits = [None]*n_models
            elites = [None]*n_models
            # ensemble - end
    
            
        # load img dataset
        if aug:
            AUGSET = f"{DATASET}/aug"
            try:
                os.makedirs(AUGSET)
            except:
                print()

            
            df = pd.read_csv(DATASET+"/dataset.csv", sep=',')
            
            # for i in idx:
            i = idx[len(idx)-1]
            image = cv2.imread(f"{DATASET}/{df.iloc[idx]['input'][i]}", cv2.IMREAD_COLOR)
            label = cv2.imread(f"{DATASET}/{df.iloc[idx]['label'][i]}", cv2.IMREAD_ANYDEPTH)
            filename =f"{AUGSET}/{num}_img.png"
            cv2.imwrite(filename, image)
            filename =f"{AUGSET}/{num}_masks.png"
            channels = [label]
            label = gray2rgb(channels[0])
            cv2.imwrite(filename, label)
            datasetwrite(num, f'{AUGSET}/dataset.csv')
            num += 1
            transformed = transform(image = image, mask = label)
            filename =f"{AUGSET}/{num}_img.png"
            cv2.imwrite(filename, transformed["image"])
            filename =f"{AUGSET}/{num}_masks.png"
            cv2.imwrite(filename, transformed["mask"])
            datasetwrite(num, f'{AUGSET}/dataset.csv')
            num += 1
        
            augdataset = read_dataset(AUGSET, indices = None)          
            train_x, train_y = augdataset.train_xy
        else:
            dataset = read_dataset(DATASET, indices=idx)
            train_x, train_y = dataset.train_xy
            # if preprocessing != None:
            #     train_x = preprocessing.call(train_x)
            # load - end
        # evolution
        y_hats = [None]*n_models
        
        for i, model in enumerate(models):
            strategies[i], elites[i], gens[i] = model.fit(train_x, train_y, elite = elites[i], gen=generations)
            y_hats[i], _ = model.predict(train_x)
            fits[i] = strategies[i].fitness.compute_one(train_y, y_hats[i])
    
        # new_var = np.var(fits)
        # print("old, new var", old_var, new_var)
        # if (abs(old_var - new_var)) <= e:
        #     generations += 10
        #     generations = min(generations, 200) 
        #     e *= 0.1
        # old_var = new_var
        
        # evolution - end

        bests_id = np.argpartition(1-np.array(fits), -1)[-1:]
        worses_id = np.argpartition(fits, -3)[-3:]
        print("fits", fits)
        print("bests_id, worses_id: ",bests_id, worses_id)  
        # evolution - end
        
        # gathering test values, for analysis
        y_hats, _ = models[bests_id[0]].predict(test_x)
        test_fit = strategies[bests_id[0]].fitness.compute_one(test_y, y_hats)
        # gathering - end
        
        
        
        # if indices: # if indices is not empty
        if len(idx) < 30:
            random.shuffle(indices)
            # active learning methods
            if method == "uncertainty_weighted":
                metric = []
                for n, img in enumerate(indices):
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
                    metric.append(val) 
                    if max_imgs <= n:
                        break     
                idx.append(indices.pop(metric.index(max(metric)))) 
                    # saving information for future analysis
            elif method == "uncertainty":
                metric = []
                for n, img in enumerate(indices):
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
                    metric.append(val) 
                    if max_imgs <= n:
                        break     
                idx.append(indices.pop(metric.index(max(metric)))) 
            elif method == "random":
                metric =  [-1,-1]
                rnd = indices.pop()
                idx.append(rnd)
                # saving information for future analysis
        # print("uncertainty: ", metric)                  
        # if len(idx) > 10:
        #     print("entrou, vai duplicar!")
        #     # idx.pop(0)
        #     metric1 = []
        #     for n, img in enumerate(idx):
        #         # loading data
        #         dataset = read_dataset(DATASET, indices=[img])
        #         x, y = dataset.train_xy
        #         # getting masks
        #         masks = [None]*n_models
        #         for i in range(n_models):
        #             masks[i], _ = models[i].predict(x)
        #         # masks - end
        #         val = 0
        #         for id1 in range(n_models):
        #             for id2 in range(i + 1, n_models):
        #                 val += count_different_pixels_weighted(masks[id1][0]["mask"], masks[j][id2]["mask"])
        #         metric1.append(val) 
        #     idx.append(metric1.index(min(metric1)))
            
        # AL - end
        if method == "uncertainty_weighted":
            eval_cost += n_models * ((len(idx)-1)*generations*(len(strategies[bests_id[0]].population.individuals)) + max_imgs)
        if method == "uncertainty":
            eval_cost += n_models * ((len(idx)-1)*generations*(len(strategies[bests_id[0]].population.individuals)) + max_imgs)
        elif method == "random":
            eval_cost += n_models * ((len(idx)-1)*generations*(len(strategies[bests_id[0]].population.individuals)))
            

        # saving information for future analysis
        # _id = fitness.index(min(fitness))
        _id = bests_id[0]
        active_nodes = models[0].parser.parse_to_graphs(elites[_id])
        # data = [run, (gen+1), eval_cost, np.min(test_fits), fitness, len(active_nodes[0]), idx]
        data = ["run", "gen", "eval_cost", "test", "train", "n_active", "used_imgs", "bests_ids", "worses_ids", "uncertainty", "image"]
        data = [run, (gen+1), eval_cost, test_fit, fits, len(active_nodes[0]), idx[0:len(idx)-1], np.argpartition(1-np.array(fits), -3)[-3:], worses_id, max(metric), idx[len(idx)-1]]
        with open(file_ensemble, 'a') as f:
                writer = csv.writer(f, delimiter = '\t')
                writer.writerow(data)
        # saving information - end
        
        
        gen += 1
        # evolution - end
                    
        y_hats, _ = models[bests_id[0]].predict(test_x)
        

        img_res = f"{RESULTS}/{gen}"
        try:
            os.makedirs(img_res)
        except:
            print()
            
        for i, y in enumerate(y_hats):
            imgs_name = f"{img_res}/run_{run}_gen_{gen-1}_img_{i}.png"
            save_prediction(imgs_name, test_v[i], y["mask"]) 

            
            
