import torch
import torch.nn as nn

#This will be an eploration of a different approach involving a game state expected winner that will evaluate current positions expected winner frequencies
#   Then it will look into the future to see what moves are most likely to make one side win

class MLP(nn.Module):
    def __init__(self,input_size=64,ouput_size=2,num_layers = 5, dropout = .1,layer_dims = []):
        super.__init__(MLP,self)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = nn.activation.ReLU()
        self.layers = [nn.Linear(self.input_size, layer_dims[0])]
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(layer_dims[i+1],layer_dims[i+2]))
        self.layer.append(nn.Linear(layer_dims[num_layers],ouput_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        for layer in self.layers:
            x = self.activation(F.Normalize(layer(x)))
        return x
        
