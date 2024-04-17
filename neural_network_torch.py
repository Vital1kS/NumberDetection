import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, learning_rate=0.01):
        super(NeuralNetwork, self).__init__()
        #Коэффициент обучения
        self.learning_rate= learning_rate
        #Слои нейросети
        self.input_layer = nn.Linear(28*28,256)
        self.output_layer = nn.Linear(256,10)
    def forward(self, x):
        x = torch.flatten(x,0)
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x