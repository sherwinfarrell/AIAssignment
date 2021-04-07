from snake_env_search import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from plot_script import plot_result
import time
import heapq

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class aStar:

    """ A* """

    


def manhattanHeuristic(problem, node):
    x1, y1 = node[0][0], node[0][1]
    x2, y2 = problem.get_apple_pos()
    print("Apple Pos" + str(problem.get_apple_pos()))
    print("ManH Dis Pos: " + str(node))
    return abs(x1 - x2) + abs(y1 - y2)

def euclideanHeuristic(problem, node):
    x1, y1 = node[0][0], node[0][1]
    x2, y2 = problem.get_apple_pos()
    return ( (x1 - x2) ** 2 + (y1 - y2) ** 2 ) ** 0.5

def getActions(env):
    startingNode = env.getSnake()
    if env.body_check_apple():
        return []

    visitedNodes = []

    pQueue = PriorityQueue()
    pQueue.push((startingNode, [], 0), 0)

    while not pQueue.isEmpty():

        currentNode, actions, prevCost = pQueue.pop()
        print("Current Node Pos")
        print(str(currentNode[0][0]) + ", " + str(currentNode[0][1]))
        print(actions)

        if not currentNode[0] in visitedNodes:
            visitedNodes.append(currentNode[0])

            if env.node_body_check_apple(currentNode):
                return actions

            for nextNode, action, cost in env.getSuccessors(currentNode):
                if not nextNode[0] in visitedNodes:
                    print("Action: " + str(action))
                    print("To Pos: " + str(nextNode[0][0]) + ", " + str(nextNode[0][1]))
                    newAction = actions + [action]
                    newCostToNode = prevCost + cost
                    heuristicCost = newCostToNode + euclideanHeuristic(env, nextNode)
                    print("ManH Dis: " + str(manhattanHeuristic(env, nextNode)))
                    print("H Cost: " + str(heuristicCost))
                    pQueue.push((nextNode, newAction, newCostToNode),heuristicCost)
        else:
            print("Current Node was in Visited")
    print("EXIT WHILE LOOP")

def gameLoop(env):

    for e in range(20):
        max_steps = 10000
        for i in range(max_steps):
            actions = getActions(env)
            for a in range(len(actions)):
                env.step(actions[a])
        env.reset()
                
            


if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

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
    actions = gameLoop(env)    