class State:
    def __init__(self, pos):
        self.agentPos = pos

    def display(self):
        print('State: ')
        print('agent position: ', self.agentPos)
        
    def __eq__(self, state):
        if state.agentPos != self.agentPos: return False
        return True

    def __hash__(self):
        return hash(self.agentPos)

class Problem:
    def __init__(self, mazeFile):
        self.__wall = '*'
        self.__agent = 'P'
        self.__dot = '.'
        self.__statesExplored = 0
        self.potential_moves = {'R': (1, 0), 'L': (-1, 0), 'D': (0, 1), 'U': (0, -1)}

        self.readMaze(mazeFile)

    def readMaze(self, mazeFile):
        self.walls = []
        self.pacman = 0
        self.dots = []
        self.xMax = 0
        self.yMax = 0
        
        f = open (mazeFile, 'r')
        if not f:
            print('File not found')
            return False        
        y = 0
        while True:
            s = list (f.readline ())
            #print (s)
            
            if s == []: break
            if s[-1] == '\n': s.pop()
            x=0
            for k in s:
                if k == self.__wall: self.walls.append ((x, y))
                if k == self.__agent: self.pacman = (x, y)
                if k == self.__dot: self.dots.append((x, y))
                x += 1
            y += 1
            self.xMax = max (self.xMax, x)
        self.yMax = y
        return True

    def isWall(self, pos):
        return pos in self.walls

    def getStartState(self):
        return State(self.pacman)

    def isValidMove(self, pos):
        x, y = pos
        return 0 <= x < self.xMax and 0 <= y < self.yMax and not self.isWall(pos)

    def isTerminal(self, state):
        return state.agentPos in self.dots

    def reward(self, state):
        if state.agentPos in self.dots: return 10
        return 0

    def transition(self, state):
        x, y = state.agentPos

        potential_moves = [((x+1, y), 'R'), ((x-1, y), 'L'),
                           ((x, y+1), 'D'), ((x, y-1), 'U')]

        moves = [(State(move), a) for move, a in potential_moves if self.isValidMove(move)]
        self.__statesExplored += 1
        return moves
