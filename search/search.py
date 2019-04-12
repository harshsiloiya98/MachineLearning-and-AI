import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a + b for (a, b) in l]
        return ''.join(modl)

    root = problem.getStartState()
    st = util.Stack()
    explored = []
    depth = 0
    st.push(Node(root, 0, 0, None, depth))
    while (not st.isEmpty()):
        current = st.pop()
        current_state = current.state
        hashed_current = convertStateToHash(current_state)
        if (problem.isGoalState(current_state)):
            return current_state
        if (hashed_current not in explored):
            explored.append(hashed_current)
            succs = problem.getSuccessors(current_state)
            depth += 1
            for succ in succs:
                st.push(Node(succ[0], succ[1], succ[2] + current.path_cost, current, depth))
    return None

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem = None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    p1 = problem.G.node[state]
    p2 = problem.G.node[problem.end_node]
    p1 = ((p1['x'], 0, 0), (p1['y'], 0, 0))
    p2 = ((p2['x'], 0, 0), (p2['y'], 0, 0))
    return util.points2distance(p1, p2)

def AStar_search(problem, heuristic = nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """
    root = problem.getStartState()
    path = []
    explored = []
    depth = 0
    pq = util.PriorityQueueWithFunction(lambda n: n.path_cost + heuristic(n.state, problem))
    pq.push(Node(root, [root], 0, None, depth))
    while (not pq.isEmpty()):
        current_node = pq.pop()
        if (problem.isGoalState(current_node.state)):
            path = current_node.action
            return path
        if (current_node.state not in explored):
            explored.append(current_node.state)
            succs = problem.getSuccessors(current_node.state)
            depth += 1
            for succ in succs:
                succnode = Node(succ[0], current_node.action + [succ[0]], succ[2] + current_node.path_cost, current_node, depth)
                pq.push(succnode)
    return path