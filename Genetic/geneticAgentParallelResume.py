from snake_env_genetic import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
import matplotlib.pyplot as plt
from keras.optimizers import Adam

import time
import keras
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from random import randint
import multiprocessing 
from pathos.multiprocessing import ProcessPool
from joblib import Parallel, delayed
import copy
import csv
import os


#This script will resume training the snakes from the last generation produced

class snakeNetwork:

    """ Deep Q Network """

    def __init__(self, env, params, weights = None):



        self.action_space = env.action_space
        self.state_space = env.state_space
        self.layer_sizes = params['layer_sizes']
        self.fitness = 0
        self.weight = weights
        self.model = None

        if weights is not None:
            self.set_weights(weights)

    def set_weights(self, weights):
        if self.model is None:
            self.build_model()
        if len(self.model.layers) != len(weights):
            print("ERROR: Weight mismatch")
            return
        for w, l in zip(weights, self.model.layers):
            #print(l == w)
            l.set_weights(w)


    def set_weights_by_layer(self, weights, layer):

        self.model.layers[layer].set_weights(weights)

    def weights(self):
        if self.model is not None:
            return [layer.get_weights() for layer in self.model.layers]
        else:
            return None

    def shape(self):
        return np.array(self.weights).shape


    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.state_space,), activation='sigmoid', kernel_initializer=keras.initializers.RandomUniform(minval=-50, maxval=50), bias_initializer=keras.initializers.RandomUniform(minval=-50, maxval=50)))
            else:
                model.add(Dense(self.layer_sizes[i], activation='sigmoid', kernel_initializer=keras.initializers.RandomUniform(minval=-50, maxval=50), bias_initializer=keras.initializers.RandomUniform(minval=-50, maxval=50)))

        self.model = model


    def act(self, state):

        act_values = self.model.predict(state)
        #print(act_values[0])

        return np.argmax(act_values[0])



def train_genetic(generations, env):

    sum_of_rewards = []
    generationAve = []
    agent = snakeNetwork(env, params)
    agents = []
    GENERATION = 1


    for _ in range(params['Population Size']):
        agents.append(snakeNetwork(env, params))
    
    for agent, snek in zip(agents,os.listdir("snakes/gen" + str(len(os.listdir("snakes"))))):

        agent.model = keras.models.load_model("snakes/" + "gen" + str(len(os.listdir("snakes"))) + "/" + snek)
        
    i = 1

    GENERATION = len(os.listdir("snakes")) + 1

    for e in range(GENERATION, generations):

        
        results, snakes_in_generation = evaluation(agents, env, GENERATION)

        agents = evolve_population(results, env,GENERATION)

    
        generationAve.append(sum(snakes_in_generation)/len(snakes_in_generation))
        GENERATION += 1

        #Give your computer a time to sit down, catch its breath, drink some water, do some stretches etc.
        time.sleep(30)
        i += 1


    return generationAve

def evaluation(agents, env, generation):    
    snakes_in_generation = []
    sum_of_rewards = []


    weightList = []
    for agent in agents:
        weightList.append(agent.weights())
        agent.model = None
        

    fitness = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(play)(agent=agents[i], weights = weightList[i], GENERATION = generation, returnScore = True) for i in range(len(agents)))
    fitness1 = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(play)(agent=agents[i], weights = weightList[i], GENERATION = generation, returnScore = True) for i in range(len(agents)))

    fitnesses = []
    scores = []

    i = 0
    for fit, agent in zip(fitness, agents):

        agent.fitness = int((fitness[i][0] + fitness1[i][0])/len(fitness))
        fitnesses.append(fit[0])
        agent.build_model()
        agent.score = fit[2]
        scores.append(fit[2])
        agent.set_weights(fit[1])
        sum_of_rewards.append(fit[0])
        snakes_in_generation.append((fit[0], agent))
        i += 1
        
    
    if not os.path.exists("snakes/gen" + str(generation)):

        os.mkdir("snakes/gen" + str(generation))

    
    i = 0
    for fit, agent in snakes_in_generation:
        agent.model.save('snakes/gen' + str(generation) + "/snake" + str(i) + ".h5")
        i += 1



    with open('fitnesses/gen' + str(generation), 'w') as f:
        
        write = csv.writer(f)
        
        write.writerow(fitnesses)


    with open('scores/gen' + str(generation), 'w') as f:
        
        write = csv.writer(f)
        
        write.writerow(scores)

    bestSnek = [fa[1] for fa in sorted(snakes_in_generation, key=lambda x: x[1].score, reverse=True)]
    print(bestSnek[0])
    bestSnek[0].model.save("savedModelsScore/gen" + str(generation) + ".h5")

    return snakes_in_generation, sum_of_rewards

@tensorflow.autograph.experimental.do_not_convert
def play(agent, weights, GENERATION = 0, returnScore = False):
    max_steps = 10000
    snake = Snake()
    agent.build_model()
    if (weights is not None):
        agent.set_weights(weights)
    state = snake.reset()
    snake.generation = GENERATION
    state = np.reshape(state, (1, snake.state_space))
    for i in range(max_steps):
        action = agent.act(state)
        prev_state = state
        next_state, reward, done, _ = snake.step(action)
        next_state = np.reshape(next_state, (1, snake.state_space))
        state = next_state

        if done:
            break
    score = len(snake.snake_body)
    agent.fitness = snake.fitness
    snake.snake.reset()
    snake.apple.reset()
    snake.score.reset()

    for body in snake.snake_body:
        body.reset()

    fitness = snake.fitness
    snake.win.clear()
    del snake
    if returnScore:
        return fitness, agent.weights(), score
    return fitness, agent.weights()

def ranked_networks(fitness_agents):
        return [fa[1] for fa in sorted(fitness_agents, key=lambda x: x[1].fitness, reverse=True)]


def evolve_population(fitness_agents,env ,generation):
    # rank by fitness
    networks = ranked_networks(fitness_agents)

    scores = [fa[0] for fa in sorted(fitness_agents, key=lambda x: x[0], reverse=True)]

    print("Average Fitness")
    print(sum(scores)/ len(scores))
    evolved = networks[:params['evolve_size']]

    weightList = []

    print("creating networks")
    randomNetworks = []
    for agent in random.sample(networks,len(evolved)):
        randomNetworks.append(agent)
    

    for agent in randomNetworks:
        weightList.append(agent.weights())
        agent.model = None

    print("Selecting Parents")

    parentWeights = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(tournament)(aWeights=weightList[random.randint(0, len(randomNetworks)-1)], bWeights=weightList[random.randint(0, len(randomNetworks)-1)], cWeights=weightList[random.randint(0, len(randomNetworks)-1)],parentA=randomNetworks[random.randint(0, len(randomNetworks)-1)], parentB=randomNetworks[random.randint(0, len(randomNetworks)-1)], parentC=randomNetworks[random.randint(0, len(randomNetworks)-1)], GENERATION = generation) for i in randomNetworks)

    parents = []
    for weight, agent in zip(parentWeights, randomNetworks):
        agent.fitness = weight[0]
        agent.build_model()
        agent.set_weights(weight[1])

        parents.append(agent)

    print("Generating Children")
    children = []

    weightList = []
    randomParentNetworks = []
    for agent in random.sample(parents,len(evolved)):
        randomParentNetworks.append(agent)

    testlist = []
    for agent in randomParentNetworks:
        weightList.append(agent.weights())
        testlist.append(snakeNetwork(env, params))


    childWeights = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(breed)(aWeights=weightList[random.randint(0, len(randomParentNetworks)-1)], bWeights=weightList[random.randint(0, len(randomParentNetworks)-1)],aInput=testlist[random.randint(0, len(testlist)-1)], bInput=testlist[random.randint(0, len(testlist)-1)], GENERATION = generation) for i in testlist)

    children = []
    for weight, agent in zip(childWeights, testlist):
        agent.fitness = weight[0]
        agent.build_model()
        agent.set_weights(weight[1])

        children.append(agent)



    print("mutating")
    weightList = []
    randomMutantNetworks = []
    for agent in range( params['evolve_size']):
        if random.random() < params["mutate_chance"]:
            randomMutantNetworks.append(networks[random.randint(0, len(networks)-1)])

    testlist = []
    for agent in randomMutantNetworks:
        weightList.append(agent.weights())
        testlist.append(snakeNetwork(env, params))


    mutantWeights = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(mutate)(network=testlist[random.randint(0, len(testlist)-1)], weights=weightList[random.randint(0, len(weightList)-1)],  generation = generation) for i in testlist)

    mutants = []
    for weight, agent in zip(mutantWeights, testlist):
        agent.fitness = weight[0]
        agent.build_model()
        agent.set_weights(weight[1])

        mutants.append(agent)


    networks = networks + children + mutants

    networks.sort(key=lambda Network: Network.fitness, reverse=True) 
    networks[0].model.save("savedModels/gen" + str(generation) + ".h5")

    emptyNets = []
    weightList = []
    for network in networks:
        weightList.append(network.weights())
        emptyNets.append(snakeNetwork(env,params))

    randints = []
    for i in range(int(0.2*len(networks))):              # More random mutations because it helps
        rand = randint(10, len(networks)-1)
        randints.append(rand)


    mutantWeights2 = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(mutate)(network=emptyNets[i], weights=weightList[i],  generation = generation) for i in randints)

    for weight, agent, rand in zip(mutantWeights, testlist, randints):
        agent.fitness = weight[0]
        agent.build_model()
        agent.set_weights(weight[1])

        networks[rand] = copy.deepcopy(agent)

    networks = networks[:params['Population Size']]
    
    print("Next gen")
    return networks

def tournament(parentA, parentB, parentC, aWeights, bWeights, cWeights,  GENERATION):

    fitnessA = play(parentA, aWeights, GENERATION)
    fitnessB = play(parentB,  bWeights, GENERATION)
    fitnessC = play(parentC, cWeights, GENERATION)



    maxscore = max(fitnessA[0], fitnessB[0], fitnessC[0])
    if maxscore == fitnessA[0]:

        return fitnessA[0], aWeights
    elif maxscore == fitnessB[0]:

        return fitnessA[0], bWeights
    else:

        return fitnessA[0], cWeights

def mutation_factor():
    return 1 + ((random.random() - 0.5) * 3 + (random.random() - 0.5)) 


def mutate(network, generation, weights):

    network.set_weights(weights)
    weights_or_biases = random.randint(0, 1)   # choosing randomly if crossover is over bias or weight/neuron/layer
    if weights_or_biases == 0:   



        layer = random.randint(0, len(network.model.layers) - 1)   
        weights = network.model.layers[layer].get_weights()

        weights[0][random.randint(0, len(weights) - 1)] = np.random.randn()

        
        network.set_weights_by_layer(weights, layer)
 
        fitness = play(network, network.weights(), generation)
        network.fitness = fitness[0]

        return fitness
    else:

        layer = random.randint(0, len(network.model.layers) - 1)   
        biases = network.model.layers[layer].get_weights()


        biases[1][random.randint(0, len(biases) - 1)] = np.random.randn()

        network.set_weights_by_layer(biases, layer)


        fitness = play(network, network.weights(), generation)
        network.fitness = fitness[0]


        return fitness



def breed(aInput, bInput, aWeights, bWeights, GENERATION):

    a = copy.deepcopy(aInput)
    b = copy.deepcopy(bInput)

    a.build_model()
    a.set_weights(aWeights)
    b.build_model()
    b.set_weights(bWeights)

    weights_or_biases = random.randint(0, 1)   
    if weights_or_biases == 0:     

        layer = random.randint(0, len(a.model.layers) - 1)   
        bNew = a.model.layers[layer].get_weights()
        aNew = b.model.layers[layer].get_weights()



        aWeights = a.model.layers[layer].get_weights()[0]
        bWeights = b.model.layers[layer].get_weights()[0]

        weightToChange = random.randint(0, len(aWeights) - 1)

        tempWeightFromA = aWeights[weightToChange] 

        aWeights[weightToChange] = bWeights[weightToChange]
        bWeights[weightToChange] = tempWeightFromA

        bNew = [bWeights, b.model.layers[layer].get_weights()[1]]
        aNew = [aWeights, a.model.layers[layer].get_weights()[1]]

        a.set_weights_by_layer(aNew, layer)
        b.set_weights_by_layer(bNew, layer)
    else:
        layer = random.randint(0, len(a.model.layers) - 1)   
        bNew = a.model.layers[layer].get_weights()
        aNew = b.model.layers[layer].get_weights()


        aBias = a.model.layers[layer].get_weights()[1]
        bBias = b.model.layers[layer].get_weights()[1]

        biasToChange = random.randint(0, len(aBias) - 1)

        tempBiasFromA = aBias[biasToChange] 

        aBias[biasToChange] = bBias[biasToChange]
        bBias[biasToChange] = tempBiasFromA

        bNew = [b.model.layers[layer].get_weights()[0], bBias]
        aNew = [a.model.layers[layer].get_weights()[0], aBias]

        a.set_weights_by_layer(aNew, layer)
        b.set_weights_by_layer(bNew, layer)


    fitnessA = play(aInput, a.weights(), GENERATION)
    fitnessB = play(bInput, b.weights(), GENERATION)



    if (fitnessA[0] > fitnessB[0]):
        return fitnessA

    else:
        return fitnessB


if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['layer_sizes'] = [12, 16, 4]
    params['Population Size'] = 1000
    params['evolve_size'] = 300
    params['mutate_chance'] = 0.7

    

    generations = 70

    env = Snake()
    sum_of_rewards = train_genetic(generations, env) 


    print(sum_of_rewards)
