from kartezio.model.evolution import KartezioES
from kartezio.population import PopulationWithElite, PopulationWithElite_mu

import random
import numpy as np

class OnePlusLambda(KartezioES):
    def __init__(self, _lambda, factory, init_method, mutation_method, fitness):
        self._mu = 1
        self._lambda = _lambda
        self.factory = factory
        self.init_method = init_method
        self.mutation_method = mutation_method
        self.fitness = fitness
        self.population = PopulationWithElite(_lambda)

    @property
    def elite(self):
        return self.population.get_elite()

    def initialization(self):
        for i in range(self.population.size):
            individual = self.init_method.mutate(self.factory.create())
            self.population[i] = individual

    def selection(self):
        new_elite, fitness = self.population.get_best_individual()
        self.population.set_elite(new_elite)

    def reproduction(self):
        elite = self.population.get_elite()
        for i in range(self._mu, self.population.size):
            self.population[i] = elite.clone()
        
    def mutation(self):
        for i in range(self._mu, self.population.size):
            self.population[i] = self.mutation_method.mutate(self.population[i])

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.call(y_true, y_pred)
        self.population.set_fitness(fitness)


class MuPlusLambda(KartezioES):
    def __init__(self, _mu, _lambda, factory, init_method, mutation_method, fitness):
        self._mu = _mu
        self._lambda = _lambda
        self.factory = factory
        self.init_method = init_method
        self.mutation_method = mutation_method
        self.fitness = fitness
        self.old_fitness_vals = 0
        self.population = PopulationWithElite_mu(self._lambda)
        self.old_population = PopulationWithElite_mu(self._lambda)
        self.idx = np.zeros(self.population.size)

    @property
    def elite(self):
        return self.population.get_elite()

    def initialization(self):
        for i in range(self.population.size):
            individual = self.init_method.mutate(self.factory.create())
            self.population[i] = individual

    def reproduction(self):
        exit("pane no sistema, alguém me desconfigurou!")

    def mutation(self):
        for i in range(self.population.size):
            self.population[i] = self.mutation_method.mutate(self.population[i])

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.call(y_true, y_pred)
        self.population.set_fitness(fitness)
        

    def selection(self): #
        tmp_population = PopulationWithElite_mu(self._lambda*2+1)        
        for i in range(self.population.size):
            tmp_population[i] = self.population[i].clone()
            
        for i in range(self.population.size):
            tmp_population[i+self.population.size] = self.old_population[i].clone()
                        
        # get the _mu best individuals
        fitness = np.array(self.population.get_fitness()+self.old_fitness_vals)
        
        # just for size/struct, information shouldn't be used !
        fit = self.population.get_fitness()
        
        best_idx = np.argsort(fitness)[0:self._lambda]
        for i, idx in enumerate(best_idx):
            self.population[i] = tmp_population[idx].clone()
            fit[i] = fitness[idx]
        self.population.set_fitness(fit)
        
        new_elite, _ = self.population.get_best_individual()
        self.population.set_elite(new_elite)
        
    def tournament(self, k):
        self.idx = np.zeros(self.population.size)
        fitness = self.population.get_fitness()
        for i in range(self.population.size):
            idx =  random.sample(range(self.population.size), k)
            fit = 1
            for j in idx:
                if fit >= fitness[j]:
                    fit = fitness[j]
                    self.idx[i] = j   
                    
        self.cloning() 
        
    def lexicase(self, cases, k):
        selected_id = []
        for _ in range(k):
            candidates = np.arange(0, self.population.size).tolist()
            # cases = list(range(len(values)))
            cases = [cases]
            random.shuffle(cases)
            
            while len(cases) > 0 and len(candidates) > 1:
                
                y_pred = self.parser.parse_population(self.strategy.population, [cases[0]])
                values = self.fitness.call(cases[0], y_pred)
                errors_for_this_case = values
                median_val = np.median(errors_for_this_case)
                median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
                
                best_val_for_case = max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = [x for x in range(len(candidates)) if values[x] >= min_val_to_survive]
                
                cases.pop(0)
            
            
            if k == 1:
                selected_id = random.choice(candidates)
            else:
                selected_id.append(random.choice(candidates))
    
        return selected_id
        
                    


    def cloning(self):
        for i in range(self.population.size):
            self.old_population[i] = self.population[i].clone()
        self.old_fitness_vals = self.population.get_fitness()
        
        # selected by tournament
        for i, idx in enumerate(self.idx):
            self.population[i] = self.old_population[int(idx)].clone()
        

