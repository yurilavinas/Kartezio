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
from typing import List, Tuple
from numena.image.color import rgb2bgr
import albumentations as A
import pandas as pd
import cv2
from numena.image.basics import image_split, image_new
from dataclasses import dataclass
from numena.image.color import gray2rgb
@dataclass
class DataItem:
    datalist: List
    shape: Tuple
    count: int
    visual: np.ndarray = None

    @property
    def size(self):
        return len(self.datalist)

def datasetwrite(i, filename):
    with open(filename, 'a', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        data = [f'{i}_img.png',f'{i}_masks.png','training']
        writer.writerow(data)

# Declare an augmentation pipeline
transform = A.Compose([
    A.GaussianBlur(p=0.1),
    A.ElasticTransform(p=0.1),
    A.GridDistortion(p=0.1),
    A.ShiftScaleRotate(p=0.1),
    A.OpticalDistortion(p=0.1),
    A.GridDistortion(p=0.1),
    A.CLAHE(p=0.1),
    A.RandomRotate90(p=0.1),
    A.Transpose(p=0.1),
    # A.CoarseDropout(max_holes=30, max_height=10, max_width=10, fill_value=64, p=0.3)
])

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
    _lambda = config["_lambda"]
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
        
        test = str([f"test_{i}" for i in range(n_models)],).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        train = str([f"train_{i}" for i in range(n_models)]).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        data = ["run", "gen", "eval_cost", *test.split(","), *train.split(","), "best_fitness", "n_active", "idx"]
        with open(file_ensemble, 'w') as f:
            writer = csv.writer(f, delimiter = '\t', quoting=csv.QUOTE_NONE)
            writer.writerow(data)
    except:
        print()

    dataset = read_dataset(DATASET, indices=None)
    train_x, train_y = dataset.train_xy
    test_x, test_y, test_v = dataset.test_xyv
    if preprocessing != None:
        test_x = preprocessing.call(test_x)
    size = len(train_x)
    gen = 0
    
    while eval_cost < maxeval:
        if gen == 0 or (restart == True and eval_cost > checkpoint): 
            if gen > 0:
                elite_name = f"{RESULTS}/restart_elite_run_{run}_gen_{gen}.json"
                model.save_elite(elite_name, dataset) 
                
                viewer = KartezioViewer(
                    model.parser.shape, model.parser.function_bundle, model.parser.endpoint
                )
                model_graph = viewer.get_graph(
                    elites, inputs=["In_1","In_2"], outputs=["out_1","out_2"]
                )
                path = f"{RESULTS}/restart_graph_model_run_{run}_gen_{gen}.png"
                model_graph.draw(path=path)
                
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
            
            elites = None
            checkpoint = checkpoint + val
            count = 0

        if count < size:
            idx = [count]
            augdataset = dataset = read_dataset(DATASET, indices=idx)
        elif method == "roulette_inc_del_bad":
            # elite = best_elites
            if count == size:
                bad_perf = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                idx = [bad_perf[0]]
            elif count > size:
                f = int((c*gen+1)**a)   
                if gen % f == 0 or len(idx) == 0: 
                    bad_perf = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()
                    while probs[bad_perf] > max(probs)*0.9:
                        bad_perf = np.random.choice(size, 1, p=np.array(probs)/sum(probs)).tolist()    
                    idx.append(bad_perf[0])
            
                    aug = True
                    print(gen, idx)
                    if aug:
                        AUGSET = f"{DATASET}/aug"
                        try:
                            os.makedirs(AUGSET)
                        except:
                            os.system(f"rm -r {AUGSET}")
                            os.makedirs(AUGSET)

                        cmd = f'cp "{DATASET}/META.json" "{AUGSET}/META.json"'
                        os.system(cmd)
                        with open(f'{AUGSET}/dataset.csv', 'w', encoding='UTF8') as f:
                            # create the csv writer
                            writer = csv.writer(f)
                            header = ["input","label","set"]
                            writer.writerow(header)
                        df = pd.read_csv(DATASET+"/dataset.csv", sep=',')
                        num = 0
                        for i in idx:
                            image = cv2.imread(f"{DATASET}/{df.iloc[idx]['input'][i]}", cv2.IMREAD_COLOR)
                            label = cv2.imread(f"{DATASET}/{df.iloc[idx]['label'][i]}", cv2.IMREAD_ANYDEPTH)
                            filename =f"{AUGSET}/{num}_img.png"
                            cv2.imwrite(filename, image)
                            filename =f"{AUGSET}/{num}_masks.png"
                            channels = [label]
                            label = gray2rgb(channels[0])
                            cv2.imwrite(filename, label)
                            datasetwrite(num, f'{AUGSET}/dataset.csv')
                            # myrange = int(10/len(idx))
                            myrange = 10
                            for j in range(myrange):
                                num += 1
                                transformed = transform(image = image, mask = label)
                                filename =f"{AUGSET}/{num}_img.png"
                                cv2.imwrite(filename, transformed["image"])
                                filename =f"{AUGSET}/{num}_masks.png"
                                cv2.imwrite(filename, transformed["mask"])
                                datasetwrite(num, f'{AUGSET}/dataset.csv')
                    
                        augdataset = read_dataset(AUGSET, indices = None)
                if len(idx) >= 10:
                    idx.pop(np.random.choice(len(idx),1)[0])
        # aug data train
        train_x, train_y = augdataset.train_xy
        if preprocessing != None:
            train_x = preprocessing.call(train_x)
        strategy, elites = model.fit(train_x, train_y, elite = elites)
        
        y_hats, _ = model.predict(train_x)
        fitness = strategy.fitness.compute_one(train_y, y_hats)
        y_hats, _ = model.predict(test_x)
        test_fits  = strategy.fitness.compute_one(test_y, y_hats)
              
        
        
        for _idx in idx:
            __idx = _idx
            probs[__idx] = fitness
            probs_inv[__idx] = 1 - fitness
        probs_uniq[idx[0]] = 1 - fitness
        
        if fitness <= train_best_ever:
            train_best_ever = fitness
            idx_best_ever = idx[0]
            test_best_ever = test_fits
            # best_elites = elites

        eval_cost += n_models * len(idx) * (len(strategy.population.individuals))
        active_nodes = model.parser.parse_to_graphs(elites)
        data = [run, (gen+1), eval_cost, test_fits, fitness, test_best_ever, len(active_nodes[0]+active_nodes[1]), idx]
        with open(file_ensemble, 'a') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)
        
        
        gen += 1
        count += 1
        print(fitness, test_fits, eval_cost)
        if eval_cost % 1000 == 0:
              
            elite_name = f"{RESULTS}/elite_run_{run}_gen_{gen}.json"
            model.save_elite(elite_name, dataset) 
            
            y_hat, _ = model.predict(test_x)
            imgs_name = f"{RESULTS}/model_run_{run}_gen_{gen}.png"
            save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
            
            viewer = KartezioViewer(
                model.parser.shape, model.parser.function_bundle, model.parser.endpoint
            )
            model_graph = viewer.get_graph(
                elites, inputs=["In_1","In_2"], outputs=["out_1","out_2"]
            )
            path = f"{RESULTS}/graph_model_run_{run}_gen_{gen}.png"
            model_graph.draw(path=path)
        
    elite_name = f"{RESULTS}/final_elite_run_{run}_gen_{gen}.json"
    model.save_elite(elite_name, dataset) 
    

    y_hat, _ = model.predict(test_x)
    imgs_name = f"{RESULTS}/final_model_run_{run}_gen_{gen}.png"
    save_prediction(imgs_name, test_v[0], y_hat[0]["mask"])
    
    viewer = KartezioViewer(
        model.parser.shape, model.parser.function_bundle, model.parser.endpoint
    )
    model_graph = viewer.get_graph(
        elites, inputs=["In_1","In_2"], outputs=["out_1","out_2"]
    )
    path = f"{RESULTS}/final_graph_model_run_{run}_gen_{gen}.png"
    model_graph.draw(path=path)