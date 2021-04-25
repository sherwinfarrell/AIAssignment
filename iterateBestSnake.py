from snake_env_genetic import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from random import randint
import os
import csv

#This script will run 100 iterations of a chosen snake. Used to generate metrics to compare it to other methods

def act(model, state):

    act_values = model.predict(state)
    return np.argmax(act_values[0])




def playGens(bestGen):

    path = "savedModels/gen"
    sum_of_rewards = []
    generationAve = []
    agents = []

    i = 1

    dirlist = (os.listdir("savedModels"))

    model = keras.models.load_model(bestGen)
    ave = []
    for e in range(0,100):
  
        ave.append(play(model,i))

        i+=1

    with open('bestSnakeScores/gen', 'w') as f:
    
        write = csv.writer(f)

        write.writerow(ave)

    print(sum(ave)/len(ave))

    return generationAve


def play(model,generation):
    max_steps = 10000
    env = Snake()
    env.set_generation(generation)
    state = env.reset()
    state = np.reshape(state, (1, env.state_space))
    for i in range(max_steps):
        action = act(model, state)
        prev_state = state
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, env.state_space))
        state = next_state
        if done:
            total = env.total
            env.win.clear()
            return total
            
        

if __name__ == '__main__':

 
    
    out = playGens("savedModels/gen39.h5")


    