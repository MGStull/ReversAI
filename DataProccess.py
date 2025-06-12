import csv
import numpy as np

class dataSet():
    def __init__(self,size=25657):
        sum = 0
        orderCords = []
        for i in range(8):
            for j in  range(8):
                orderCords.append((i+1,j+1))
        with open("DataSet/othello_dataset.csv") as file:
            csvreader = csv.reader(file)
            for i,row in enumerate(csvreader):
                if(i>=size):
                    break
                self.games.append(self.processRow(row,orderCords))
    #black moves first in Othello
    def processRow(self,row,orderCords):
        if row[1] == '-1':
            winner = -1
        elif row[1] == '1':
            winner = 1
        else:
            winner = 0
        game = [row[2][i:i+2]for i in range(0,len(row[2]),2)]
        boards = np.zeros((len(game),65))
        moves =[]
        boards[0][orderCords.index((4,4))] = 1
        boards[0][orderCords.index((4,5))]= -1
        boards[0][orderCords.index((5,5))] = 1
        boards[0][orderCords.index((5,4))]= -1 

        for i,move in enumerate(game):
            if i == len(boards)-1:
                break
            x,y = ord(move[0])-96,ord(move[1])-48
            moves.append((x,y))
            if i > 0:
                boards[i+1]=boards[i].copy()
            boards[i+1][orderCords.index((x,y))] = (-1)**(i+1)
            boards[i][64]=winner
        print(len(boards))
        yOneHot = [orderCords.index(i) for i in moves]
        return yOneHot,boards
