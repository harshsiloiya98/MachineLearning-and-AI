import sys
import numpy as np

if __name__ == "__main__":
    p = 1
    filename = sys.argv[1]
    if (len(sys.argv) == 3):
        p = float(sys.argv[2])
    grid = []
    s = []
    e = []
    r = 0
    with open(filename, 'r') as f:
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
    grid = np.array(grid)
    # removing the boundaries
    grid = np.delete(grid, 0, axis = 0)
    grid = np.delete(grid, -1, axis = 0)
    grid = np.delete(grid, 0, axis = 1)
    grid = np.delete(grid, -1, axis = 1)
    (r, c) = grid.shape
    numStates = r * c
    numActions = 4
    discount = 0.9
    # storing the grid in row-major form
    start = c * (s[0] - 1) + s[1] - 1
    end = c * (e[0] - 1) + e[1] - 1
    grid_rm = np.reshape(grid, (numStates, 1))
    print("numStates", numStates)
    print("numActions", numActions)
    print("start", start)
    print("end", end)
    for act in range(numActions):
        for i in range(r):
            for j in range(c):
                source = i * c + j
                if (grid_rm[source][0] == 3 or grid_rm[source][0] == 1):
                    continue
                if (act == 0):
                    # N transition
                    if (i == 0):
                        #continue
                        dest = source
                        print("transition", source, act, dest, -1, p)
                    else:
                        dest = (i - 1) * c + j
                        if (grid_rm[dest][0] == 1 or grid_rm[dest][0] == 2):
                            print("transition", source, act, dest, -1, p)
                        elif (grid_rm[dest][0] == 3):
                            print("transition", source, act, dest, 1, p)
                        else:
                            print("transition", source, act, dest, 0, p)
                elif (act == 1):
                    # E transition
                    if (j == c - 1):
                        #continue
                        dest = source
                        print("transition", source, act, dest, -1, p)
                    else:
                        dest = i * c + j + 1
                        if (grid_rm[dest][0] == 1 or grid_rm[dest][0] == 2):
                            print("transition", source, act, dest, -1, p)
                        elif (grid_rm[dest][0] == 3):
                            print("transition", source, act, dest, 1, p)
                        else:
                            print("transition", source, act, dest, 0, p)
                elif (act == 2):
                    # S transition
                    if (i == r - 1):
                        #continue
                        dest = source
                        print("transition", source, act, dest, -1, p)
                    else:
                        dest = (i + 1) * c + j
                        if (grid_rm[dest][0] == 1 or grid_rm[dest][0] == 2):
                            print("transition", source, act, dest, -1, p)
                        elif (grid_rm[dest][0] == 3):
                            print("transition", source, act, dest, 1, p)
                        else:
                            print("transition", source, act, dest, 0, p)
                elif (act == 3):
                    # W transition
                    if (j == 0):
                        #continue
                        dest = source
                        print("transition", source, act, dest, -1, p)
                    else:
                        dest = i * c + j - 1
                        if (grid_rm[dest][0] == 1 or grid_rm[dest][0] == 2):
                            print("transition", source, act, dest, -1, p)
                        elif (grid_rm[dest][0] == 3):
                            print("transition", source, act, dest, 1, p)
                        else:
                            print("transition", source, act, dest, 0, p)
    print("discount ", discount)