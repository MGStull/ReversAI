import csv
import numpy as np

class dataSet():
    def __init__(self,size=25657):
        sum = 0
        self.coordTOindex = []
        self.games = []
        self.x_data = []
        self.y_data = []
        for i in range(8):
            for j in  range(8):
                self.coordTOindex.append((i+1,j+1))

        with open("DataSet/othello_dataset.csv") as file:
            csvreader = csv.reader(file)
            for i,row in enumerate(csvreader):
                if(i>=size):
                    break
                self.games.append(self.processRow(row))
        self.flatten()
    #black moves first in Othello
    def processRow(self,row):
        if row[1] == '-1':
            winner = -1
        elif row[1] == '1':
            winner = 1
        else:
            winner = 0
        game = [row[2][i:i+2]for i in range(0,len(row[2]),2)]
        boards = np.zeros((len(game),65))
        moves =[]
        boards[0][self.coordTOindex.index((4,4))] = 1
        boards[0][self.coordTOindex.index((4,5))]= -1
        boards[0][self.coordTOindex.index((5,5))] = 1
        boards[0][self.coordTOindex.index((5,4))]= -1 

        for i,move in enumerate(game):
            if i == len(boards)-1:
                break
            x,y = ord(move[0])-96,ord(move[1])-48
            moves.append((x,y))
            if i > 0:
                boards[i+1]=boards[i].copy()
            #A piece has been place on the board of color -1 then it is 1's turn
            boards[i][self.coordTOindex.index((x,y))] = (-1)**(i+1)
            #expected Move is a move of color ((-1)**(i+1))*(-1) to move[i+1]
            boards[i][64] = ((-1)**(i+1))*(-1)
        return moves,boards

    def flatten(self):
        for moves, boards in self.games:
            for i in range(len(moves) - 1):  # avoid out-of-bound
                self.x_data.append(boards[i].reshape(65, 1))
                x, y = moves[i + 1]
                label = self.coordTOindex.index((x, y))  # label ∈ [0, 63]
                self.y_data.append(label)
