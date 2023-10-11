# from kartezio.endpoint import EndpointThreshold
# from kartezio.apps.segmentation_interactive import create_segmentation_model_interactive
from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.callback import CallbackVerbose
import random
import numpy as np
import csv
import time
from kartezio.preprocessing import SelectChannels

class   active_learning():
    def __init__(self, cfg):

        framework = cfg["framework"]
        config = cfg["variables"]
        self.generations = config["generations"]
        self._lambda = config["_lambda"]
        
        self.n_models = config["n_models"]
        self.verbose = config["verbose"]
        self.img_t = config["img_t"] 
        self.frequency = config["frequency"]
        self.ask_user = config["ask_user"]
        self.iter = config["al_iter"]
        self.user_t = config["user_t"] # threshold to move from AL and ask user for data
        self.eval_cost = 0
        self.method = config["method"]
        # based on data
        self.DATASET = framework["DATASET"]        
        self.filename = framework["filename"]
        if self.DATASET == "/tmpdir/lavinas/cellpose":
            CHANNELS = [1, 2]
            self.preprocessing = SelectChannels(CHANNELS)
            dataset = read_dataset(self.DATASET, indices=None)
        elif self.DATASET == "/tmpdir/lavinas/ssi":
            self.meta_filename = f"META_rgb.json"
            dataset = read_dataset(self.DATASET, indices=None, filename=self.filename, meta_filename=self.meta_filename, preview=False)
        self.OUTPUT = framework["OUTPUT"]

        
        x, _ = dataset.train_xy
        
        self.indices_c = list(range(0, len(x))) 
        self.indices_nc = []#list(range(int(imgs_c), 2*int(imgs_nc)))
        self.run = -1
        # framework
        self.elites = [None]*self.n_models
        self.fitness = [None]*self.n_models
        self.y_model = [None]*self.n_models # trash
        self.strategies = [None]*self.n_models
        self.models = [None]*self.n_models
        self.history = []
        self.select_id = None
        self.lvls = list()
        
        for i in range(self.n_models):
            # self.models[i] = create_instance_segmentation_model(
            #     self.generations,
            #     self._lambda,
            #     inputs=inputs,
            #     nodes=nodes,
            #     outputs=outputs,
            #     fitness=fitness_fun,
            #     endpoint=endpoint
            # )
            self.models[i] = create_instance_segmentation_model(
                self.generations,
                self._lambda,
                inputs=config["input"],
                outputs=config["output"],
            )

        
        random.shuffle(self.indices_c)
        self.idx = self.indices_c.pop()
        self.lvls.append(self.idx)
        
        
        if len(self.indices_nc) > 0:
            self.indices = self.indices_c + self.indices_nc
        else:
            self.indices = self.indices_c
        
        if self.verbose:
            print("Starting with data from index: ", self.lvls)
            print("Un-used data (index): ", self.indices, "\n")
            
        for i in range(self.n_models):
            self.history.append([])

        self.status = ['empty']*self.n_models


    def init_model(self):
        ## create function here
        for n, model in enumerate(self.models):
            model.clear()
            verbose = CallbackVerbose(frequency=self.frequency)
            # save = CallbackSave_interactive(self.OUTPUT, dataset, gen, n, run, frequency=self.frequency)
            # callbacks = [verbose, save]
            callbacks = [verbose]
            # self.workdir = save.workdir
            if callbacks:
                for callback in callbacks:
                        callback.set_parser(model.parser)
                        model.attach(callback)
             
        np.random.seed(42^self.run + int(time.time()))
    
    def calc_disagreement(self, dis_idxs):
        # from small imgs not used
        # we get the performance of the models
        # and select the img that lead to more diverse values
        disagreement = list()
        for i, index in enumerate(dis_idxs):
            # 
            if self.DATASET == "/tmpdir/lavinas/cellpose":
                dataset_active = read_dataset(self.DATASET, indices=[index])
            elif self.DATASET == "/tmpdir/lavinas/ssi":
                dataset_active = read_dataset(self.DATASET, indices=[index], filename=self.filename, meta_filename=self.meta_filename, preview=False)
                
            train_x_active, train_y_active = dataset_active.train_xy
            train_x_active = self.preprocessing.call(train_x_active)
                        
            model_res = list()
            for j in range(self.n_models):
                for k in range(j+1, self.n_models):
                    y1, _ = self.models[0].parser.parse(self.elites[j], train_x_active)
                    y2, _ = self.models[0].parser.parse(self.elites[k], train_x_active)
                    ##################################################################
                    # y = ([{"labels": np.append(y1[0]["labels"], y2[0]["labels"], axis=0)}])
                    # train_y = [np.append(train_y_active[0][0], train_y_active[0][0], axis=0)]
                    ##################################################################
                    v1 = self.strategies[0].fitness.compute_one(train_y_active, y1)
                    v2 = self.strategies[0].fitness.compute_one(train_y_active, y2)
                    model_res.append(np.abs(v1-v2))
                    # model_res.append(v1)
            # get the mean of the abs(diff)
            disagreement.append(np.mean(model_res))
        
        return disagreement
    

    def add_img(self):  

        if self.verbose:
            print("\n Active learning (adding data)...")
            print('disagreement between imgs: ', self.disagreement)
            print("waiting data (indices): ", self.indices)
 

        # get the idx of the img that leads to a higher mean of the abs(diff) 
        if self.method == "disagreement":
            id_ = self.disagreement.index(max(self.disagreement))                 
        elif self.method == "random":
            id_ = np.random.randint(0, len(self.indices))
        if self.verbose:
                print("idx: ",id_)
                print("indices to use: ",self.indices)
                print("Adding data from index: ",self.indices[id_], "\n")
                
        # adding the img selected by the active learning approach 
        # to lvls so that we can repeat the train with the old img + this selected
        idx = self.indices.pop(id_)
        self.lvls.append(idx)

    def rm_img(self):
        # random.shuffle(self.lvls) 
        disagreement = self.calc_disagreement(self.lvls) 
        disagreement = disagreement / max(disagreement)    
        for i, l in enumerate(self.lvls):       
            if disagreement[i] < self.img_t:
                if self.verbose:
                    print("\n Active learning (removing data)...")
                    print('disagreement between imgs: ', disagreement)
                    print("disagreement is below threshold...")
                    print("indices in use: ",self.lvls)
                    print("img to remove: ", l)
        
                # adding the img selected by the active learning approach 
                # to lvls so that we can repeat the train with the old img + this selected 
                self.indices.append(l)
                self.lvls.remove(l)
                break

        if len(self.lvls) == 0:
            random.shuffle(self.indices_c)
            self.idx = self.indices_c.pop()
            self.lvls.append(self.idx)
            if len(self.indices_nc) > 0:
                self.indices = self.indices_c + self.indices_nc
            else:
                self.indices = self.indices_c
            

        if self.verbose:
            print("new indices in use: ",self.lvls, "\n")

    # def saving_test(self, file, run, cycle):
    #     DATASET = read_dataset(self.DATASET, indices=None, filename=self.filename_test, meta_filename=self.meta_filename, preview=False)
    #     test_x, test_y = DATASET.test_xy

    #     idx = self.fitness.index(min(self.fitness))
        
    #     sep='_'
    #     fitness=[None]*self.n_models
    #     for i, model in enumerate(self.models):  
    #         y_hats, _ = model.parser.parse(self.elites[i], test_x)
    #         res = self.strategies[0].fitness.compute_one(test_y, y_hats)
    #         fitness[i] = res
        
    #     self.eval_cost += (self.generations * len(self.lvls) * self._lambda  + len(self.indices)) * self.n_models
    #     data = [run, cycle, self.eval_cost, *fitness, fitness[idx], *self.fitness, np.sum(self.disagreement), sep.join(map(str,self.lvls)), self.method]]
    #     with open(file, 'a') as f:
    #         writer = csv.writer(f, delimiter = '\t')
    #         writer.writerow(data)

    #     del test_x, test_y
    
    def saving_test(self, file, cycle):
        # DATASET = read_dataset(self.DATASET, indices=None, filename=self.filename_test, meta_filename=self.meta_filename, preview=False)
        if self.DATASET == "/tmpdir/lavinas/cellpose":
            dataset = read_dataset(self.DATASET, indices=None)
        elif self.DATASET == "/tmpdir/lavinas/ssi":
            dataset = read_dataset(self.DATASET, indices=None, filename=self.filename, meta_filename=self.meta_filename, preview=False)
            
        test_x, test_y = dataset.test_xy
        test_x = self.preprocessing.call(test_x)
        
        idx = self.fitness.index(min(self.fitness))
        
        sep='_'
        
        fitness=[None]*self.n_models
        for i, model in enumerate(self.models):  
            y_hats, _ = model.parser.parse(self.elites[i], test_x)
            res = self.strategies[0].fitness.compute_one(test_y, y_hats)
            fitness[i] = res
        if self.method == "disagreement":            
            self.eval_cost += self.generations * len(self.lvls) * self._lambda * self.n_models + len(self.indices) * self.n_models
        elif self.method == "random":
            self.eval_cost += self.generations * len(self.lvls) * self._lambda * self.n_models 
            self.disagreement = 0
        elif self.method == "inv":            
            self.eval_cost += self.generations * len(self.lvls) * self._lambda * self.n_models + len(self.indices) * self.n_models
           
        data = [self.run, cycle, self.eval_cost, fitness[idx], np.sum(self.disagreement), sep.join(map(str,self.lvls)), self.method, *fitness, *self.fitness, ]
        with open(file, 'a') as f:
            writer = csv.writer(f, delimiter = '\t')
            writer.writerow(data)

        del test_x, test_y

    def update_stats(self):
        # what to do with the models?
        # we get running the selected
        # we get running the one with the highest fitness
        # we copy the selected to the models that are left (empty) 
        # if the selected is the one with the highest fitness
        # we copy the selected_highest to the models that are left (empty) 
        best_id = self.fitness.index(min(self.fitness))
        #### 
        self.status = ['empty']*self.n_models

        if self.select_id != None:

            if best_id == self.select_id:
                self.status[self.select_id] = 'selected_highest'
            else:
                self.status[best_id] = 'best'
                self.status[self.select_id] = 'selected'
        
        for i in range(len(self.history)):
            self.history[i].append(self.status[i])

        for i, stat in enumerate(self.status):
            for j in range(len(self.status)):
                if i != j:
                    if stat == 'selected_highest':
                        if self.status[j] == 'empty':
                            self.elites[j] = self.elites[i]
                            self.status[j] = 'branching'
                        elite_id = i
                    elif stat == 'best':
                        elite_id = i
                    elif stat == 'selected':
                        if self.status[j] == 'empty':
                            self.elites[j] = self.elites[i]
                            self.status[j] = 'branching'
                    else: #it's empty
                        continue
        
        if self.verbose:
            print("history",self.history)

    def user_input(self):
        while True: 
            try:
                self.select_id = int(input("Enter the number of the image (1 to n): \n"))
            except:
                self.select_id = int(input("Enter the number of the image (1 to n): \n")) 
            if 1 <= self.select_id <= self.n_models:
                self.select_id -= 1
                break
