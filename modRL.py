from snake_env import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':


    epsilon = 1
    lr = 0.025
    gamma = 0.95 #Penalty for q learning 
    memory = deque(maxlen=3000)
    model = Sequential()
    model.add(Dense(248, input_shape=(12,), activation='relu'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(248, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))

    batch_size = 64
    epsilon_min = 0.01
    epsilon_decay = 0.995


    episodes = 5
    game = Snake()

    scores = []
    for e in range(episodes):
        state = game.reset()
        state = np.reshape(state, (1, 12))
        score = 0
        max_steps = 10000
        for i in range(max_steps):
            if np.random.rand() <= epsilon:
                action = random.randrange(4)
            else:
                action = np.argmax(model.predict(state)[0])
            prev_state = state
            next_state, reward, done, _ = game.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 12))
            memory.append((state, action, reward, next_state, done))
            state = next_state
            # agent.replay()

            if len(memory)>= batch_size:
                
                minibatch = random.sample(memory, batch_size)
                states = np.array([i[0] for i in minibatch])
                actions = np.array([i[1] for i in minibatch])
                rewards = np.array([i[2] for i in minibatch])
                next_states = np.array([i[3] for i in minibatch])
                dones = np.array([i[4] for i in minibatch])

                states = np.squeeze(states)
                next_states = np.squeeze(next_states)

                targets = rewards + gamma*(np.amax(model.predict_on_batch(next_states), axis=1))*(1-dones)
                targets_full = model.predict_on_batch(states)

                ind = np.array([i for i in range(batch_size)])
                targets_full[[ind], [actions]] = targets

                model.fit(states, targets_full, epochs=1, verbose=0)

                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

            if done:
                print(f'Previous state: {str(prev_state)}')
                print(f'episode: {e+1}/{episodes}, score: {score}')
                break
        scores.append(score)
     
    
    plt.plot(range(1,episodes+1),scores)
    plt.xlabel("Episodes", fontsize = 14)
    plt.ylabel("Scores", fontsize = 14)
    plt.title("Scores For Every Episode")
    plt.show()

    


