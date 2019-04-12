import sys
import numpy as np

def value_iteration(numStates, endStates, numActions, transitions, rewards, discount, epsilon = 1e-16):
    values = np.zeros((numStates, 1))
    tmp = np.zeros((numActions, ))
    num_iterations = 0
    while (True):
        prev_values = np.copy(values)
        error = 0.
        for source in range(numStates):
            for action in range(numActions):
                t_curr = np.reshape(transitions[source][action], (numStates, 1))
                r_curr = np.reshape(rewards[source][action], (numStates, 1))
                r_curr = r_curr + discount * prev_values
                res = np.matmul(np.transpose(t_curr), r_curr)
                tmp[action] = res[0][0]
            idx = np.argmax(tmp)
            if (source in endStates):
                values[source][0] = 0.
            else:
                values[source][0] = tmp[idx]
            error = max(error, abs(values[source][0] - prev_values[source][0]))
        num_iterations += 1
        if (error <= epsilon):
            return values, num_iterations

def get_optimum_policy(numStates, endStates, numActions, transitions, rewards, discount):
    pi = np.zeros((numStates, ))
    tmp = np.zeros((numActions, ))
    for s in range(numStates):
        if (s in endStates):
            pi[s] = -1
        else:
            for act in range(numActions):
                t_curr = np.reshape(transitions[s][act], (numStates, 1))
                r_curr = np.reshape(rewards[s][act], (numStates, 1))
                r_curr = r_curr + discount * values
                res = np.matmul(np.transpose(t_curr), r_curr)
                tmp[act] = res[0][0]
            pi[s] = np.argmax(tmp)
    return pi

if __name__ == "__main__":
    filename = sys.argv[1]
    numStates = 0
    numActions = 0
    start = 0
    end = []
    discount = 0.
    transitions = []
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            type_var = tokens[0]
            if (type_var == "numStates"):
                numStates = int(tokens[1])
            elif (type_var == "numActions"):
                numActions = int(tokens[1])
                transitions = np.zeros((numStates, numActions, numStates), dtype = float)
                rewards = np.zeros((numStates, numActions, numStates), dtype = float)
            elif (type_var == "start"):
                start = int(tokens[1])
            elif (type_var == "end"):
                for i in range(1, len(tokens)):
                    end.append(int(tokens[i]))
            elif (type_var == "transition"):
                s1 = int(tokens[1])
                ac = int(tokens[2])
                s2 = int(tokens[3])
                r = float(tokens[4])
                p = float(tokens[5])
                transitions[s1][ac][s2] = p
                rewards[s1][ac][s2] = r
            else:
                discount = float(tokens[2])
    values, t = value_iteration(numStates, end, numActions, transitions, rewards, discount)
    values = np.round(values, 11)
    policy = get_optimum_policy(numStates, end, numActions, transitions, rewards, discount)
    for i in range(numStates):
        print(values[i][0], int(policy[i]))
    print(t)