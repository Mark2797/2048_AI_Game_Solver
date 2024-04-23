from random import randint
from BaseAI_3 import BaseAI
import math
import time

start = 0

class PlayerAI(BaseAI):
    def __init__(self):
        self.possibleNewTiles = [2, 4]
        self.probability = 0.9
        self.time_limit = 0.1

        self.smoothnessWeight = 0.1
        self.monotonicityWeight = 1.0
        self.emptyWeight = 2.7
        self.maxWeight = 1.0
        self.distanceWeight = 10.0

    def evaluate(self, grid):

        objective_func_choice = 'Benjamin'

        if (objective_func_choice == 'MarkN1'):
            # Mark's Objective Function
            # N-1 Pattern
            pattern = [[16, 15, 14, 13], [15, 14, 13, 12], [14, 13, 12, 11], [13, 12, 11, 10]]
            # S Pattern
            # pattern = [[16, 15, 14, 13], [9, 10, 11, 12], [8, 7, 6, 5], [1, 2, 3, 4]]
            evaluation = 0
            for i in range(grid.size):
                for j in range(grid.size):
                    evaluation += grid.map[i][j] * pattern[i][j]
            return evaluation / (grid.size * grid.size - len(grid.getAvailableCells()))
        elif (objective_func_choice == 'MarkS'):
            # Mark's Objective Function
            # N-1 Pattern
            # pattern = [[16, 15, 14, 13], [15, 14, 13, 12], [14, 13, 12, 11], [13, 12, 11, 10]]
            # S Pattern
            pattern = [[16, 15, 14, 13], [9, 10, 11, 12], [8, 7, 6, 5], [1, 2, 3, 4]]
            evaluation = 0
            for i in range(grid.size):
                for j in range(grid.size):
                    evaluation += grid.map[i][j] * pattern[i][j]
            return evaluation / (grid.size * grid.size - len(grid.getAvailableCells()))
        else:
            # Benjamin's Objective Function
            board = grid.map

            weighted_sum = self.N1_pattern_weight(board)
            chain_score = self.chain_score(board)
            score = weighted_sum + chain_score
            for j in range(len(board)):
                for i in range(len(board[0])):
                    if (board[i][j] != 0):
                        frac = 	1.0 / (self.move_distance(board, i, j) * self.value_similarity(board, i , j) + 0.01)
                        penalty = 3 * self.diag_penalty(board, i, j) + 4 * self.loc_penalty(board)

                        score += 0.5 * frac - 0.2 * penalty
            return score


    # New function: add up values of tiles that are in a chain where the next tile is twice the previous
    def chain_score(self, board):
        output = 0
        for row in range(4):
            for col in range(4):
                chain_continues = True
                row_temp = row
                col_temp = col
                while(chain_continues):
                    # Make array of all horizontal and vertical neighbors (see value_similarity() for logic behind list comprehension)
                    shifts = [-1, 0, 1]
                    neighbor_coords = [[row_temp + shifts[i], col_temp + shifts[j]] for j in range(3) for i in range(3) if ((shifts[i] != 0 and shifts[j] == 0) or (shifts[i] == 0 and shifts[j] != 0)) and (row_temp + shifts[i] >= 0 and col_temp + shifts[j] >= 0) and (row_temp + shifts[i] < 4 and col_temp + shifts[j] < 4)]
                    neighbor_vals = [board[i][j] for (i,j) in neighbor_coords]

                    # Check whether the chain continues and move on to next in chain if so
                    if (2 * board[row_temp][col_temp] in neighbor_vals) and (board[row_temp][col_temp] != 0):
                        output += 1
                        next_in_chain = neighbor_coords[neighbor_vals.index(2 * board[row_temp][col_temp])]
                        (row_temp, col_temp) = (next_in_chain[0], next_in_chain[1])
                    else:
                        chain_continues = False

        return output


    # New function: find the move-distance between a tile and all other tiles on the board of the same value
    def move_distance(self, board, row, col):
        tile = board[row][col]

        if tile == 0:
            return 0

        dist = 0
        for j in range(4):
            for i in range(4):
                # if the ith, jth tile has the same value as the tile we're focused on (and isn't that tile itself),
                if board[i][j] == tile and not (i == row and j == col):
                    # # if tile of same value shares a row or column, then it will only need one move (ignoring the presence of other tiles)
                    # dist += 1 + (i != row and j != col)	

                    dist += abs(row - i) + abs(col - j)
        return dist

    # New function: find the value-similarity for a specific tile
    def value_similarity(self, board, row, col):
        # Find the coordinates of all possible neighbors (filter out the following edge cases:    tile itself                           and  negative coordinates                           and  coordinates outside the board)
        shifts = [-1, 0, 1]
        neighbor_coords = [[row + shifts[i], col + shifts[j]] for j in range(3) for i in range(3) if (shifts[i] != 0 or shifts[j] != 0) and (row + shifts[i] >= 0 and col + shifts[j] >= 0) and (row + shifts[i] < 4 and col + shifts[j] < 4)]

        output = 0
        for n in neighbor_coords:
            if (board[n[0]][n[1]] != 0):
                output += abs(board[n[0]][n[1]] - board[row][col])
        return output

    # New function: find weight for each coordinate according to the N1 Pattern and sum weighted tile values
    def N1_pattern_weight(self, board):
        N1_pattern_weights = [[16, 15, 14, 13],[15, 14, 13, 12],[14, 13, 12, 11],[13, 12, 11, 10]]
        sum = 0
        for j in range(4):
            for i in range(4):
                sum += board[i][j] * N1_pattern_weights[i][j]
        return sum

    # New function: calculate a penalty based on diagonally adjacent tiles
    def diag_penalty(self, board, row, col):
        if (board[row][col] == 0):
            return 0

        shifts = [-1, 0, 1]
        neighbor_coords = [[row + shifts[i], col + shifts[j]] for j in range(3) for i in range(3) if (shifts[i] != 0 or shifts[j] != 0) and (row + shifts[i] >= 0 and col + shifts[j] >= 0) and (row + shifts[i] < 4 and col + shifts[j] < 4)]
        
        output = 0
        for n in neighbor_coords:
            if (board[n[0]][n[1]] == board[row][col]) and (abs(n[0] - row) == 1 and abs(n[1] - col) == 1):
                output += 2
        return output

    # New function: calculate a penalty based on location of certain-valued tiles
    def loc_penalty(self, board):
        middle = [(1,1), (1,2), (2,1), (2,2)]
        smalls = [2, 4, 8, 16]
        
        output = 0
        for j in range(4):
            for i in range(4):
                if (board[i][j] == 0):
                    continue

                if ((i,j) in middle) and not (board[i][j] in smalls):	# penalize if there are large values in the middle coords
                    output += 1
                if not ((i,j) in middle) and (board[i][j] in smalls):	# penalize if there are small values in the outer coords
                    output += 1
        return output


    @staticmethod
    def distance(grid, max_tile):
        dis = None

        for x in range(grid.size):

            if dis:
                break

            for y in range(grid.size):
                if max_tile == grid.map[x][y]:

                    if max_tile < 1024:
                        dis = -((abs(x - 0) + abs(y - 0)) * max_tile)
                    else:
                        dis = -((abs(x - 0) + abs(y - 0)) * (max_tile / 2))
                    break

        return dis

    @staticmethod
    def get_max_value(max_tile, empty_cells):
        return math.log(max_tile) * empty_cells / math.log(2)

    @staticmethod
    def monotonicity(grid):

        totals = [0, 0, 0, 0]

        for x in range(3):

            currentIndex = 0
            nextIndex = currentIndex + 1

            while nextIndex < 4:
                while nextIndex < 4 and grid.map[x][nextIndex] == 0:
                    nextIndex += 1

                if nextIndex >= 4:
                    nextIndex -= 1

                currentValue = math.log(grid.map[x][currentIndex]) / math.log(2) if grid.map[x][currentIndex] else 0
                nextValue = math.log(grid.map[x][nextIndex]) / math.log(2) if grid.map[x][nextIndex] else 0

                if currentValue > nextValue:
                    totals[0] += currentValue + nextValue
                elif nextValue > currentValue:
                    totals[1] += currentValue - nextValue

                currentIndex = nextIndex
                nextIndex += 1

        for y in range(3):

            currentIndex = 0
            nextIndex = currentIndex + 1

            while nextIndex < 4:
                while nextIndex < 4 and grid.map[nextIndex][y] == 0:
                    nextIndex += 1

                if nextIndex >= 4:
                    nextIndex -= 1

                currentValue = math.log(grid.map[currentIndex][y]) / math.log(2) if grid.map[currentIndex][y] else 0
                nextValue = math.log(grid.map[nextIndex][y]) / math.log(2) if grid.map[nextIndex][y] else 0

                if currentValue > nextValue:
                    totals[2] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[3] += currentValue - nextValue

                currentIndex = nextIndex
                nextIndex += 1

        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    @staticmethod
    def smoothness(grid):

        smoothness = 0

        for x in range(grid.size):
            for y in range(grid.size):
                s = float('infinity')

                if x > 0:
                    s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x - 1][y] or 2)))
                if y > 0:
                    s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x][y - 1] or 2)))
                if x < 3:
                    s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x + 1][y] or 2)))
                if y < 3:
                    s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x][y + 1] or 2)))

                smoothness -= s

        return smoothness

    def get_new_tile(self):
        if randint(0, 99) < 100 * self.probability:
            return self.possibleNewTiles[0]
        else:
            return self.possibleNewTiles[1]

    def search(self, grid, alpha, beta, depth, player):

        if time.perf_counter() - start > self.time_limit:
            return self.evaluate(grid), -1, True

        if depth == 0:
            return self.evaluate(grid), -1, False

        if player:

            best_score, best_move = alpha, None

            positions = grid.getAvailableMoves()

            if len(positions) == 0:
                return self.evaluate(grid), None, False

            for position in positions:

                new_grid = grid.clone()
                new_grid.move(position)

                score, move, timeout = self.search(new_grid, alpha, beta, depth - 1, False)

                if score > best_score:
                    best_score, best_move = score, position

                if best_score >= beta:
                    break

                if best_score > alpha:
                    alpha = best_score

            return best_score, best_move, False

        else:

            best_score, best_move = beta, None

            cells = grid.getAvailableCells()

            if len(cells) == 0:
                return self.evaluate(grid), None, False

            for cell in cells:

                value = self.get_new_tile()

                new_grid = grid.clone()
                new_grid.setCellValue(cell, value)

                score, move, timeout = self.search(new_grid, alpha, beta, depth - 1, True)

                if score < best_score:
                    best_score, best_move = score, None

                if best_score <= alpha:
                    break

                if best_score < beta:
                    beta = best_score

            return best_score, None, False

    def iterative(self, grid):
        global start
        # for depth 4
        best_score, depth = -float('infinity'), 4
        
        # for depth 3
        # best_score, depth = -float('infinity'), 3

        start = time.perf_counter()

        while True:

            score, move, timeout = self.search(grid, -float('infinity'), float('infinity'), depth, True)

            if timeout:
                break

            if score > best_score:
                best_move, best_score = move, score

            depth += 1

        return best_move

    def getMove(self, grid):
        return self.iterative(grid)
