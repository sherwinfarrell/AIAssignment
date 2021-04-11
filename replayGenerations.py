from snake_env_genetic import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras
from plot_script import plot_result
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from random import randint

def act(model, state):



    act_values = model.predict(state)
    #print(act_values[0])

    return np.argmax(act_values[0])




def playGens(genStart, genEnd, env):

    path = "savedModels/gen"
    sum_of_rewards = []
    generationAve = []
    agents = []

    i = 1
    for e in range(genStart,genEnd):
        env.generation = i 

        genPath = path + str(i) 
        model = keras.models.load_model(genPath)
        play(model, env)
        i+=1
    #print(len(agents))
    #print(snakes_in_generation)
    return generationAve

def evaluation(agents, env):    
    snakes_in_generation = []
    sum_of_rewards = []
    for agent in agents:
        

        score = 0
        fitness, snake = play(agent,env)

        sum_of_rewards.append(fitness)
        snakes_in_generation.append(snake)

    return snakes_in_generation, sum_of_rewards

def play(model, env):
    max_steps = 1000
    state = env.reset()
    state = np.reshape(state, (1, env.state_space))
    for i in range(max_steps):
        action = act(model, state)
        # print(action)
        prev_state = state
        next_state, reward, done, _ = env.step(action)
        #score += reward
        next_state = np.reshape(next_state, (1, env.state_space))
        #agent.remember(state, action, reward, next_state, done)
        state = next_state
        #  if params['batch_size'] > 1:
            #agent.replay()
        if done:
            #print(f'final state before dying: {str(prev_state)}')
            #print(f'episode: {e+1}/{episode}, score: {score}')

            break


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
    
    out = playGens(1,11,env)
    #return [NeuralNetwork((8, 10, 4)) for _ in range(self.pop_size)]
    #return [TFNN((8, 10, 4)) for _ in range(self.pop_size)]


    