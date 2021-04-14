from snake_env_genetic import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
import matplotlib.pyplot as plt
from keras.optimizers import Adam

from plot_script import plot_result
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

class DQN:

    """ Deep Q Network """

    def __init__(self, env, params, weights = None):



        self.action_space = env.action_space
        self.state_space = env.state_space
        self.epsilon = params['epsilon'] 
        self.gamma = params['gamma'] 
        self.batch_size = params['batch_size'] 
        self.epsilon_min = params['epsilon_min'] 
        self.epsilon_decay = params['epsilon_decay'] 
        self.learning_rate = params['learning_rate']
        self.layer_sizes = params['layer_sizes']
        self.fitness = 0
        #self.memory = deque(maxlen=2500)
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

        #print(l == w)
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
        #model.add(Dense(self.action_space, activation='softmax'))
        #model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):



        act_values = self.model.predict(state)
        #print(act_values[0])

        return np.argmax(act_values[0])


    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets




def train_dqn(episode, env):

    sum_of_rewards = []
    generationAve = []
    agent = DQN(env, params)
    agents = []
    GENERATION = 1
    for _ in range(params['Population Size']):
        agents.append(DQN(env, params))
    i = 1
    for e in range(episode):


        results, snakes_in_generation = evaluation(agents, env, GENERATION)

        agents = evolve_population(results, env,GENERATION)

    
        generationAve.append(sum(snakes_in_generation)/len(snakes_in_generation))
        GENERATION += 1
        print(GENERATION)
        i += 1
    #print(len(agents))
    #print(snakes_in_generation)
    return generationAve

def evaluation(agents, env, generation):    
    snakes_in_generation = []
    sum_of_rewards = []


    weightList = []
    #fitness, snake  = [pool.apply(play, args=(agent)) for agent in agents]
    for agent in agents:
        weightList.append(agent.weights())
        agent.model = None
        

    fitness = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(play)(agent=agents[i], weights = weightList[i], GENERATION = generation, returnScore = True) for i in range(len(agents)))
    fitness1 = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(play)(agent=agents[i], weights = weightList[i], GENERATION = generation, returnScore = True) for i in range(len(agents)))
    fitness2 = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(play)(agent=agents[i], weights = weightList[i], GENERATION = generation, returnScore = True) for i in range(len(agents)))
    fitness3 = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(play)(agent=agents[i], weights = weightList[i], GENERATION = generation, returnScore = True) for i in range(len(agents)))

    fitnesses = []
    scores = []

    i = 0
    for fit, agent in zip(fitness, agents):

        agent.fitness = int((fitness[i][0] + fitness1[i][0] + fitness2[i][0] + fitness3[i][0])/len(fitness))
        fitnesses.append(fit[0])
        agent.build_model()
        agent.score = fit[2]
        scores.append(fit[2])
        agent.set_weights(fit[1])
        sum_of_rewards.append(fit[0])
        snakes_in_generation.append((fit[0], agent))
        i += 1

    with open('fitnesses/gen' + str(generation), 'w') as f:
        
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(fitnesses)


    with open('scores/gen' + str(generation), 'w') as f:
        
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(scores)

    bestSnek = [fa[1] for fa in sorted(snakes_in_generation, key=lambda x: x[1].score, reverse=True)]
    print(bestSnek[0])
    bestSnek[0].model.save("savedModelsScore/gen" + str(generation) + ".h5")

    return snakes_in_generation, sum_of_rewards

@tensorflow.autograph.experimental.do_not_convert
def play(agent, weights, GENERATION = 0, returnScore = False):
    max_steps = 1000
    snake = Snake()
    agent.build_model()
    if (weights is not None):
        agent.set_weights(weights)
    state = snake.reset()
    snake.generation = GENERATION
    state = np.reshape(state, (1, snake.state_space))
    for i in range(max_steps):
        action = agent.act(state)
        # print(action)
        prev_state = state
        next_state, reward, done, _ = snake.step(action)
        #score += reward
        next_state = np.reshape(next_state, (1, snake.state_space))
        #agent.remember(state, action, reward, next_state, done)
        state = next_state
        #  if params['batch_size'] > 1:
            #agent.replay()
        if done:
            #print(f'final state before dying: {str(prev_state)}')
            #print(f'episode: {e+1}/{episode}, score: {score}')

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
    # print("Top Scores")
    # print(sum(scores[:params['evolve_size']]) / params['evolve_size'])
    # print("Bottom Scores")
    # print(sum(scores[params['evolve_size']:]) / params['evolve_size'])

  
 

    #print(scores)
    #print("Top Scores")    
    #print(scores[:params['evolve_size']])
    print("Average Score")
    print(sum(scores)/ len(scores))
    # # keep pick top [:evolve_size]
    evolved = networks[:params['evolve_size']]



    # print('init_evolved:',len(evolved))
    #print(evolved)
    # # randomly pick from rest
    # # for network in networks[self.evolve_size:]:
    #     # if random.random() < self.rand_select:
    #         # evolved.append(network)
    #print('post_random_add:',len(evolved))

    # # randomly pick 2 from [evolve_size:] and breed remaining pop_size - len(evolved)
    #print(evolved)
    # print("Selecting Parents")
    # parents = []
    # for agent in evolved:
    #     parent = tournament(env,
    #         networks[random.randint(0, len(networks)-1)],
    #         networks[random.randint(0, len(networks)-1)],
    #         networks[random.randint(0, len(networks)-1)],
    #         generation)
    #     parents.append(parent)

    weightList = []
    #fitness, snake  = [pool.apply(play, args=(agent)) for agent in agents]

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
        #print(weight[1] == None)
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
        testlist.append(DQN(env, params))


    childWeights = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(breed)(aWeights=weightList[random.randint(0, len(randomParentNetworks)-1)], bWeights=weightList[random.randint(0, len(randomParentNetworks)-1)],aInput=testlist[random.randint(0, len(testlist)-1)], bInput=testlist[random.randint(0, len(testlist)-1)], GENERATION = generation) for i in testlist)

    children = []
    for weight, agent in zip(childWeights, testlist):
        agent.fitness = weight[0]
        agent.build_model()
        agent.set_weights(weight[1])

        children.append(agent)

    # for agent in evolved:
    #     parentA = random.randint(0, params['evolve_size'] - 1)
    #     parentB = random.randint(0, params['evolve_size'] -1)
        
    #     children.append(breed(parents[parentA], parents[parentB], generation))
    #     #evolved.append(self.breed(networks[parentA], networks[parentB]))
    # print('post_breed:',len(evolved))


    weightList = []
    randomMutantNetworks = []
    for agent in range( params['evolve_size']):
        if random.random() < params["mutate_chance"]:
            randomMutantNetworks.append(networks[random.randint(0, len(networks)-1)])

    testlist = []
    for agent in randomMutantNetworks:
        weightList.append(agent.weights())
        testlist.append(DQN(env, params))


    mutantWeights = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(mutate)(network=testlist[random.randint(0, len(testlist)-1)], weights=weightList[random.randint(0, len(weightList)-1)],  generation = generation) for i in testlist)
    print("mutating")
    mutants = []
    for weight, agent in zip(mutantWeights, testlist):
        agent.fitness = weight[0]
        agent.build_model()
        agent.set_weights(weight[1])

        mutants.append(agent)

    # # # mutate subset of evolved
    # print("mutating")
    # mutants = []
    # for i in range(0, params['evolve_size']):

    #     if random.random() < params["mutate_chance"]:

    #         mutation = mutate(networks[random.randint(0, len(networks)-1)],generation)
    #         mutants.append(mutation)
    #         #evolved.append(GANeuralNetwork(network.dimensions))
    


    networks = networks + children + mutants

    networks.sort(key=lambda Network: Network.fitness, reverse=True) 
    networks[0].model.save("savedModels/gen" + str(generation) + ".h5")

    emptyNets = []
    weightList = []
    for network in networks:
        weightList.append(network.weights())
        emptyNets.append(DQN(env,params))

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

    # for network in networks:
    #     print(network.fitness)
    networks = networks[:params['Population Size']]
    
    print("Next gen")
    return networks

def tournament(parentA, parentB, parentC, aWeights, bWeights, cWeights,  GENERATION):

    # print("aweights: " + str(aWeights == None))
    # print("bWeights: " + str(bWeights == None))
    # print("cWeights: " + str(cWeights == None))

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
    #print("mutation factor")
    #print(1 + ((random.random() - 0.5) * 3 + (random.random() - 0.5)))
    return 1 + ((random.random() - 0.5) * 3 + (random.random() - 0.5)) 


def mutate(network, generation, weights):

    network.set_weights(weights)
    weights_or_biases = random.randint(0, 1)   # choosing randomly if crossover is over bias or weight/neuron/layer
    if weights_or_biases == 0:   



        layer = random.randint(0, len(network.model.layers) - 1)   
        weights = network.model.layers[layer].get_weights()

        #  bNew = np.array(a.model.layers[layer].get_weights()[0], b.model.layers[layer].get_weights()[1])
        # aNew = np.array(b.model.layers[layer].get_weights()[0], a.model.layers[layer].get_weights()[1])
        #print(type(aNew))
        weights[0][random.randint(0, len(weights) - 1)] = np.random.randn()

        
        network.set_weights_by_layer(weights, layer)
        # output = DQN(env, params)

        # fitness = play(output, networkCopy.weights(), generation)
        # output.fitness = fitness[0]
        # output.build_model()
        # output.set_weights(fitness[1])
        fitness = play(network, network.weights(), generation)
        network.fitness = fitness[0]

        return fitness
    else:

        layer = random.randint(0, len(network.model.layers) - 1)   
        biases = network.model.layers[layer].get_weights()

        #  bNew = np.array(a.model.layers[layer].get_weights()[0], b.model.layers[layer].get_weights()[1])
        # aNew = np.array(b.model.layers[layer].get_weights()[0], a.model.layers[layer].get_weights()[1])
        #print(type(aNew))
        #print(biases[1][random.randint(0, len(biases) - 1)])

        biases[1][random.randint(0, len(biases) - 1)] = np.random.randn()
        #print(biases[1][random.randint(0, len(biases) - 1)])
        network.set_weights_by_layer(biases, layer)
        # output = DQN(env, params)

        # fitness = play(output,networkCopy.weights(), generation)
        # output.fitness = fitness[0]
        # output.build_model()
        # output.set_weights(fitness[1])
        fitness = play(network, network.weights(), generation)
        network.fitness = fitness[0]


        return fitness
    # #print(weights)
    
    # #print(weights)
    # fitnessA, snakeA = play(a,env)
    # fitnessB, snakeB = play(b,env)

    # if (fitnessA > fitnessB):
    #     return a

    # else:
    #     return b

# return evolved
def breed(aInput, bInput, aWeights, bWeights, GENERATION):

    a = copy.deepcopy(aInput)
    b = copy.deepcopy(bInput)

    a.build_model()
    a.set_weights(aWeights)
    b.build_model()
    b.set_weights(bWeights)

    weights_or_biases = random.randint(0, 1)   # choosing randomly if crossover is over bias or weight/neuron/layer
    if weights_or_biases == 0:     

        layer = random.randint(0, len(a.model.layers) - 1)   
        bNew = a.model.layers[layer].get_weights()
        aNew = b.model.layers[layer].get_weights()

        #  bNew = np.array(a.model.layers[layer].get_weights()[0], b.model.layers[layer].get_weights()[1])
        # aNew = np.array(b.model.layers[layer].get_weights()[0], a.model.layers[layer].get_weights()[1])
        #print(type(aNew))

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

        #  bNew = np.array(a.model.layers[layer].get_weights()[0], b.model.layers[layer].get_weights()[1])
        # aNew = np.array(b.model.layers[layer].get_weights()[0], a.model.layers[layer].get_weights()[1])
        #print(type(aNew))

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

    #print(weights)
    
    #print(weights)

    fitnessA = play(aInput, a.weights(), GENERATION)
    fitnessB = play(bInput, b.weights(), GENERATION)



    if (fitnessA[0] > fitnessB[0]):
        return fitnessA

    else:
        return fitnessB


if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 0.2
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .2
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [12, 16, 4]
    params['Population Size'] = 1000
    params['evolve_size'] = 300
    params['mutate_chance'] = 0.7


    results = dict()
    ep = 50

    # for batchsz in [1, 10, 100, 1000]:
    #     print(batchsz)
    #     params['batch_size'] = batchsz
    #     nm = ''
    #     params['name'] = f'Batchsize {batchsz}'
    env_infos = {'States: only walls':{'state_space':'no body knowledge'}, 'States: direction 0 or 1':{'state_space':''}, 'States: coordinates':{'state_space':'coordinates'}, 'States: no direction':{'state_space':'no direction'}}

    # for key in env_infos.keys():
    #     params['name'] = key
    #     env_info = env_infos[key]
    #     print(env_info)
    #     env = Snake(env_info=env_info)
    env = Snake()
    
    sum_of_rewards = train_dqn(ep, env) 
    #return [NeuralNetwork((8, 10, 4)) for _ in range(self.pop_size)]
    #return [TFNN((8, 10, 4)) for _ in range(self.pop_size)]


    print(sum_of_rewards)

    results[params['name']] = sum_of_rewards
    
    plot_result(results, direct=True, k=20)
    