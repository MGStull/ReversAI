import DataProccess as DP
import ReversAI as AI
import Reversi as RSI
import MCTS as MCTS

NN = AI.Neural_Network([
    [65,300],
    [300,200],
    [200,128],
    [128,64]
    ],.1)

data = DP.dataSet()
x_data = data.x_data
y_data = data.y_data
for i in range(20):
    epoch = AI.Epoch(1000,x_data,y_data,64)
    print(epoch.kmc(NN))