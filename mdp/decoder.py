import sys
import random
import numpy as np

if __name__ == "__main__":
    policy = []
    values = []
    grid = []
    s = []
    e = []
    r = 0
    p = 1
    directions = {0:'N', 1:'E', 2:'S', 3:'W'}
    gridfilename = sys.argv[1]
    vpfilename = sys.argv[2]
    if (len(sys.argv) == 4):
        p = float(sys.argv[3])
    with open(gridfilename, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            tokens = list(map(int, tokens))
            if (2 in tokens):
                c = tokens.index(2)
                s = (r, c)
            if (3 in tokens):
                c = tokens.index(3)
                e = (r, c)
            grid.append(tokens)
            r += 1
    with open(vpfilename, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if (len(tokens) == 1):
                break
            values.append(float(tokens[0]))
            policy.append(int(tokens[1]))
    grid = np.array(grid)
    # removing the boundaries
    grid = np.delete(grid, 0, axis = 0)
    grid = np.delete(grid, -1, axis = 0)
    grid = np.delete(grid, 0, axis = 1)
    grid = np.delete(grid, -1, axis = 1)
    (r, c) = grid.shape
    i = s[0] - 1
    j = s[1] - 1
    start = c * i + j
    end = c * (e[0] - 1) + e[1] - 1
    idx = start
    moves = []
    while (idx != end):
        if (p == 1):
            move = policy[idx]
        else:
            prb = [0., 0., 0., 0.]
            validity = [False, False, False, False]
            move = policy[idx]
            num_valid_moves = 0
            if (i > 0 and grid[i - 1][j] != 1):
                num_valid_moves += 1
                validity[0] = True
            if (j < c - 1 and grid[i][j + 1] != 1):
                num_valid_moves += 1
                validity[1] = True
            if (i < r - 1 and grid[i + 1][j] != 1):
                num_valid_moves += 1
                validity[2] = True
            if (j > 0 and grid[i][j - 1] != 1):
                num_valid_moves += 1
                validity[3] = True
            if (num_valid_moves > 0):
                rem_p = (1 - p) / num_valid_moves
                for k in range(4):
                    if (k != move and validity[k]):
                        prb[k] = rem_p
                    elif (k == move):
                        prb[k] = p + rem_p
            else:
                prb[move] = 1.
            actual_move = np.random.choice([0, 1, 2, 3], size = 1, p = prb)
            move = actual_move[0]
        if (move == 0):
            i -= 1
        elif (move == 1):
            j += 1
        elif (move == 2):
            i += 1
        else:
            j -= 1
        idx = c * i + j
        moves.append(directions[move])
    for move in moves:
        print(move, end = ' ')