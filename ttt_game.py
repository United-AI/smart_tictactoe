import numpy as np
import random as rd
import re

class TTT:
    def __init__(self,tile1="X",tile2="O",empty=" "):
        self.tile1=tile1
        self.tile2=tile2
        self.empty=empty
        self.board=np.array([[self.empty for i in range(3)] for i in range(3)])
        self.turn=1
        self.begin=rd.randint(0,1)        
    def put(self,x,y):
        if x>2 or y>2:
            return False
        if self.board[x,y]!=self.empty:
            return False
        self.board[x,y]=self.tile1 if self.turn%2==self.begin else self.tile2
        self.turn+=1
        return True
    def evaluate(self):
        unique,_=np.unique(self.board, return_counts=True)
        if not (self.empty in unique):
            return "Nobody"
        for i in range(3):
            if (self.board[0,i]==self.board[1,i]==self.board[2,i]!=self.empty):
                return self.board[0,i]
            if (self.board[i,0]==self.board[i,1]==self.board[i,2]!=self.empty):
                return self.board[i,0]
        if (self.board[0,0]==self.board[1,1]==self.board[2,2]!=self.empty):
                return self.board[0,0]
        if (self.board[0,2]==self.board[1,1]==self.board[2,0]!=self.empty):
                return self.board[0,2]
        return False
    def showBoard(self):
        print(f""" Y
 |---|---|---|
1| {self.board[0,0]} | {self.board[1,0]} | {self.board[2,0]} |
 |---|---|---|
2| {self.board[0,1]} | {self.board[1,1]} | {self.board[2,1]} |
 |---|---|---|
3| {self.board[0,2]} | {self.board[1,2]} | {self.board[2,2]} |
 |---|---|---> X
   1   2   3""")
if __name__=="__main__":
    ttt=TTT()
    while 1:
        ttt.showBoard()
        print(f"\nturn: {ttt.tile1 if ttt.turn%2==ttt.begin else ttt.tile2}")
        while 1:
            if ttt.put(*[int(i)-1 for i in input("x,y:").split(",")]):
                break
            else:
                print("incorrect input, try again")
        if ttt.evaluate():
            ttt.showBoard()
            print(ttt.evaluate(),"won")
            break
