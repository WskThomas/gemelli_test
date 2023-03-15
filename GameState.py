SIZE = 4
WIN_SIZE = 4

#used for no prior information victory conditions
pointGenerators = [(x,y,z) for x in range(SIZE) for y in range(SIZE) for z in range(SIZE) if x == 0 or y == 0 or z == 0]  
directions = [1,0,-1]
vectors = [(i,j,k) for i in directions for j in directions for k in directions if i!=0 or j!=0 or k!=0 ]
half_vectors = vectors[0:len(vectors)//2]
mirror_vectors = list(reversed(vectors[len(vectors)//2 :]))

#This dict allows us to know which points to check for a win condition, whenever the last move is on a given point
pointsToCheckDic = {}
for x,y,z in [(x,y,z) for x in range(SIZE) for y in range(SIZE) for z in range(SIZE)]:
    p0 = (x,y,z)
    pointsToCheckDic[p0] = []
    for vf, vb in zip(half_vectors, mirror_vectors):
        lastF = p0
        lastB= p0
        fPointList = []
        bPointList = []
        for _ in range(WIN_SIZE-1):
            lastF = tuple(map(sum, zip(lastF, vf)))
            lastB = tuple(map(sum, zip(lastB, vb)))
            fPointList += [lastF]
            bPointList += [lastB]
        for i in range(WIN_SIZE):
            pointList = bPointList[:WIN_SIZE - i]+[p0]+fPointList[:i]
            inboundPoints = [p for p in pointList if all(c>=0 and c<SIZE for c in p)]
            if len(inboundPoints) == WIN_SIZE :
                pointsToCheckDic[p0] = pointsToCheckDic[p0] + [inboundPoints]

class GameState:
    Grid = [] # Grid[x][y][z] is None if contains no peg, else 0 or 1 corresponding to the player number
    IsPlayerZeroTurn = True
    LastMove = None
    MoveCount = 0

    def __init__(self) -> None:
        self.Grid = [[[None for k in range(SIZE)] for j in range(SIZE)] for i in range(SIZE)]    
        self.IsPlayerZeroTurn = True
        self.LastMove = None
        self.MoveCount = 0

    ###
    # returns a list 3-tuples with the coordinates of all legal moves
    ###
    def getPossibleMoves(self) -> list:
        return [(x,y, self.Grid[x][y].index(None)) for x in range(SIZE) for y in range(SIZE) if None in self.Grid[x][y]] #None means a peg spot is empty

    def checkEnd(self) -> bool : 
        return self.getWinner() is not None or self.MoveCount == SIZE**3

    def getWinner(self) -> int:
        if self.LastMove is None :
            return None
        p = self.LastMove
        lastMoveValue = self.Grid[p[0]][p[1]][p[2]]
        if any( all(V == lastMoveValue for V in [self.Grid[x][y][z] for (x,y,z) in inboundPoints]) for inboundPoints in pointsToCheckDic[p]) :
            return lastMoveValue

    ###
    # generally slower win condition checker
    ###
    def getWinnerAutoCompute(self) -> int:
        if self.LastMove is None :
            return None
        for vf, vb in zip(half_vectors, mirror_vectors):
            p0 = self.LastMove
            pointList = [p0]
            lastF = p0
            lastB = p0
            for _ in range(WIN_SIZE-1):
                lastF = tuple(map(sum, zip(lastF, vf)))
                lastB = tuple(map(sum, zip(lastB, vb)))
                pointList += [lastF, lastB]
            inboundPoints = [p for p in pointList if all(c>=0 and c<SIZE for c in p)]
            lastMoveValue = self.Grid[p0[0]][p0[1]][p0[2]]
            if len(inboundPoints) >= WIN_SIZE and all(V == lastMoveValue for V in [self.Grid[x][y][z] for (x,y,z) in inboundPoints]) :
                return lastMoveValue 
        return None

    ###
    # generally slower win condition checker but can work without knowing what the last move was
    ###
    def getWinnerWithNoPriorInfo(self) -> int:
        for (p,v) in [(point,vector) for point in pointGenerators for vector in vectors] :
            lastP = p
            pointList = [p]
            for _ in range(WIN_SIZE-1):
                lastP = tuple(map(sum, zip(lastP, v)))
                pointList += [lastP]
            
            edgeValue = self.Grid[p[0]][p[1]][p[2]]
            if edgeValue is not None and all(all(c>=0 and c<SIZE for c in p) for p in pointList) and all(edgeValue == V for V in [self.Grid[x][y][z] for (x,y,z) in pointList]) :
                return edgeValue 
        return None
    
    def playLegalMove(self, move : tuple) -> None:
        self.LastMove = (move[0],move[1],move[2])
        if(self.Grid[move[0]][move[1]].index(None) != move[2] ):
            raise Exception("ILLEGAL MOVE")
        self.Grid[move[0]][move[1]][move[2]] = 0 if self.IsPlayerZeroTurn else 1
        self.IsPlayerZeroTurn = not self.IsPlayerZeroTurn
        self.MoveCount += 1