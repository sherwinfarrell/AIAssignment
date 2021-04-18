from snake_env import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from plot_script import plot_result
import time




if __name__ == '__main__':

    name = None
    epsilon = 1
    gamma = .95
    batch_size = 128
    epsilon_min = .01
    epsilon_decay = .995
    lr = 0.00025
    memory = deque(maxlen=3000)

    model = Sequential()
    model.add(Dense(248, input_shape=(12,), activation='relu'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(248, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))

    episodes = 20
    env = Snake()
    scores = []
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 10000
        for i in range(max_steps):
            action = None
            if np.random.rand() <= epsilon:
                action = random.randrange(4)
            else:
                act_values = model.predict(state)
                action =  np.argmax(act_values[0])
            # print(action)
            prev_state = state
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            memory.append((state, action, reward, next_state, done))
            state = next_state
            if batch_size > 1:
                if len(memory) < batch_size:
                    print("doning nothing")
                else:
                    minibatch = random.sample(memory, batch_size)
                    states = np.array([i[0] for i in minibatch])
                    actions = np.array([i[1] for i in minibatch])
                    rewards = np.array([i[2] for i in minibatch])
                    next_states = np.array([i[3] for i in minibatch])
                    dones = np.array([i[4] for i in minibatch])

                    states = np.squeeze(states)
                    next_states = np.squeeze(next_states)
                    next_states_pred = model.predict_on_batch(next_states)
                    targets = rewards + gamma*(np.amax(next_states_pred, axis=1))*(1-dones)
                    targets_full = model.predict_on_batch(states)

                    ind = np.array([i for i in range(batch_size)])
                    targets_full[[ind], [actions]] = targets

                    model.fit(states, targets_full, epochs=1, verbose=0)
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episodes: {e+1}/{episodes}, score: {score}')
                break
        scores.append(score)
    
    plt.plot(range(1,episodes+1),scores)
    plt.xlabel("episodess", fontsize = 14)
    plt.ylabel("Scores", fontsize = 14)
    plt.title("Scores For Every episodes")
    plt.show()
    
    