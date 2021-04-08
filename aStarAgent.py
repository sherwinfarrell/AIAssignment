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

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        #print("List length: " + str(len(self.list)))
        return len(self.list) == 0

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0 


def manhattanHeuristic(problem, node):
    x1, y1 = node[0][0], node[0][1]
    x2, y2 = problem.get_apple_pos()
    return abs(x1 - x2) + abs(y1 - y2)

def euclideanHeuristic(problem, node):
    x1, y1 = node[0][0], node[0][1]
    x2, y2 = problem.get_apple_pos()
    return ( (x1 - x2) ** 2 + (y1 - y2) ** 2 ) ** 0.5

# A*
def getActions(env):
    startingNode = env.getSnake()
    if env.body_check_apple():
        return []

    visitedNodes = []

    pQueue = PriorityQueue()
    pQueue.push((startingNode, [], 0), 0)

    while not pQueue.isEmpty():

        currentNode, actions, prevCost = pQueue.pop()
        #print(currentNode)
        if not currentNode[0] in visitedNodes:
            visitedNodes.append(currentNode[0])

            if env.node_body_check_apple(currentNode):
                print("Path found")
                if not checkDeadendDFS(env, currentNode):
                    return actions

            for nextNode, action, cost in env.getSuccessors(currentNode):
                if not nextNode[0] in visitedNodes:
                    newAction = actions + [action]
                    newCostToNode = prevCost + cost
                    heuristicCost = newCostToNode + euclideanHeuristic(env, nextNode)
                    pQueue.push((nextNode, newAction, newCostToNode),heuristicCost)
    print("EXIT WHILE LOOP")
    
# Breadth First Search
def checkDeadend(env, node):
    start = node
    actions = []
    visited = []
    visited.append(start)
    fringe = Queue()
    fringe.push((start, actions))

    while not fringe.isEmpty():
        current, actions = fringe.pop()
        #print("Action Length: " + str(len(actions)))
        if len(actions) == 30:
            print("Path good")
            print(actions)
            return False
        
        for successor, action, stepCost in env.getSuccessors(current):
            if successor not in visited:
                visited.append(successor)
                fringe.push((successor, actions + [action]))
    print("Deadend")
    return True

# Depth First Search
def checkDeadendDFS(env, node):
    start = node
    actions = []
    visited = []
    visited.append(start)
    fringe = Stack()
    fringe.push((start, actions))
    
    d = 500

    while not fringe.isEmpty():
        current, actions = fringe.pop()
        print("Action Length: " + str(len(actions)))
        if len(actions) == 30:
            print("Path good")
            print(actions)
            return False
        
        d -= 1
        if d < 0:
            return True
        
        for successor, action, stepCost in env.getSuccessors(current):
            if successor not in visited:
                visited.append(successor)
                fringe.push((successor, actions + [action]))
    print("Deadend")
    return True

def survive(env):
    start = env.getSnake()
    actions = []
    visited = []
    visited.append(start)
    fringe = Queue()
    fringe.__init__()
    fringe.push((start, actions))
    depth = 5

    while not fringe.isEmpty():
        current, actions = fringe.pop()
        print("Survival")
        print(len(actions))
        if len(actions) > depth:
            if not checkDeadendDFS(env, current):
                return actions
        
        for successor, action, stepCost in env.getSuccessors(current):
            if successor not in visited:
                visited.append(successor)
                fringe.push((successor, actions + [action]))
    return actions

def gameLoop(env):

    for e in range(20):
        max_steps = 10000
        for i in range(max_steps):
            actions = getActions(env)
            if actions is None:
                actions = survive(env)
            #TODO write BFS like checkDeadend if len(actions) == 0. This BFS should return the longest possible path up till a certain depth, then try A* again
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