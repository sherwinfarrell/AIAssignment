from snake_env_search import Snake

import heapq
import matplotlib.pyplot as plt


"""
Taken from Berkley AI Pacman Graph Search Questions
"""
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

"""
Taken from Berkley AI Pacman Graph Search Questions
"""
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


"""
Taken from Berkley AI Pacman Graph Search Questions
"""
def manhattanHeuristic(problem, node):
    x1, y1 = node[0][0], node[0][1]
    x2, y2 = problem.get_apple_pos()
    return abs(x1 - x2) + abs(y1 - y2)

"""
Taken from Berkley AI Pacman Graph Search Questions
"""
def euclideanHeuristic(problem, node):
    x1, y1 = node[0][0], node[0][1]
    x2, y2 = problem.get_apple_pos()
    return ( (x1 - x2) ** 2 + (y1 - y2) ** 2 ) ** 0.5

# A*
def getActions(env):
    startingNode = env.getSnake()
    if env.node_body_check_apple(startingNode):
        return []

    visitedNodes = []

    pQueue = PriorityQueue()
    pQueue.push((startingNode, [], 0), 0)

    while not pQueue.isEmpty():

        currentNode, actions, prevCost = pQueue.pop()

        if not currentNode[0] in visitedNodes:
            visitedNodes.append(currentNode[0])

            if env.node_body_check_apple(currentNode):
                if not checkDeadendDFS(env, currentNode, len(startingNode)):
                    return actions

            for nextNode, action, cost in env.getSuccessors(currentNode):
                if not nextNode[0] in visitedNodes:
                    newAction = actions + [action]
                    newCostToNode = prevCost + cost
                    heuristicCost = newCostToNode + euclideanHeuristic(env, nextNode)
                    pQueue.push((nextNode, newAction, newCostToNode),heuristicCost)


# Depth First Search forward checking
def checkDeadendDFS(env, node, maxdepth):
    start = node
    actions = []
    visited = []
    visited.append(start)
    fringe = Stack()
    fringe.push((start, actions))
    
    d = 1000

    while not fringe.isEmpty():
        current, actions = fringe.pop()
        if len(actions) == maxdepth:
            return False
        
        d -= 1
        if d < 0:
            return True
        
        for successor, action, stepCost in env.getSuccessors(current):
            if successor not in visited:
                visited.append(successor)
                fringe.push((successor, actions + [action]))
    return True

def survive(env):
    start = env.getSnake()
    actions = []
    visited = []
    visited.append(start)
    fringe = Stack()
    fringe.__init__()
    fringe.push((start, actions))
    depth = 0
    max_step = 600

    while not fringe.isEmpty():
        current, actions = fringe.pop()
        
        if len(actions) > depth:
            if not checkDeadendDFS(env, current, len(start)):
                return actions
        
        max_step -= 1
        if max_step < 0:
            return []
        for successor, action, stepCost in env.getSuccessors(current):
            if successor not in visited:
                visited.append(successor)
                fringe.push((successor, actions + [action]))
    return actions

def gameLoop(env):
    score = 0
    max_steps = 10000
    for i in range(max_steps):
        actions = getActions(env)
        if actions is None or not actions:
            actions = survive(env)
        if actions is None or not actions:
            break
        for a in range(len(actions)):
            env.step(actions[a])
    score = env.total
    env.clear()
    return score
            
            


if __name__ == '__main__':

    scores = []
    ep = 100
    
    
    for e in range(ep):
        print("Episode: " + str(e))
        env = Snake()
        score = gameLoop(env)
        scores.append(score)
    
    plt.plot(range(1,ep+1),scores)
    plt.xlabel("Episodes", fontsize = 14)
    plt.ylabel("Scores", fontsize = 14)
    plt.title("Graph Search - Scores For Every Episode")
    plt.show()
    
