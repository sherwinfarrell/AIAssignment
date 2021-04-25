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

#This script will play the best performing snake by generation, so you can see how it improves across time

def act(model, state):

    act_values = model.predict(state)

    return np.argmax(act_values[0])


def playGens(genStart, genEnd):

    path = "savedModels/gen"
    sum_of_rewards = []
    generationAve = []
    agents = []

    i = 1

    dirlist = (os.listdir("savedModels"))
    
    for e in range(genStart,len(os.listdir("savedModels"))):
        try:
            model = keras.models.load_model("savedModels" + "/" + "gen" + str(i) + ".h5")
            play(model, i)
        except:
            print("snake does not exist for gen" + str(i))
        i+=1

    return generationAve

def play(model, generation):
    max_steps = 10000
    env = Snake()
    state = env.reset()
    env.set_generation(generation)
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


    
    out = playGens(1,45)


    