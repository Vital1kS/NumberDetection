import pandas as pd
from neural_network_torch import NeuralNetwork
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

learning_rate = 0.01
mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_net = NeuralNetwork(learning_rate=learning_rate).to(device=device)
labels=[[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],
        ]

def train(n_iter, n_gen):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(main_net.parameters(), lr=learning_rate)
    max_accuracy = 0
    for gen in range(n_gen):
        for iter in range(n_iter):
            index = np.random.randint(0,60000)
            optimizer.zero_grad()
            input_data = torch.tensor(mnist_train.iloc[index].iloc[1:].to_numpy().flatten()/255).float().to(device=device)
            target = torch.tensor(labels[mnist_train.iloc[index].iloc[0]]).to(device=device)
            output = main_net(input_data).to(device=device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        accuracy = validate(main_net)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            #save_model(main_net, "model2.pt")
        
def validate(model):
    correct=0
    for index in range(10000):
        if np.argmax(a=model(torch.tensor(
            mnist_test.iloc[index].iloc[1:].to_numpy()
            .flatten()/255).float().to(device=device))
            .cpu().detach().numpy()) == mnist_test.iloc[index].iloc[0]:
                correct +=1
    print(f'Accuracy: {correct/100}%')
    return correct/100

def save_model(model, name):
    torch.save(model.state_dict(),name)

def load_model(model,name):
    model.load_state_dict(torch.load(name))
    model.eval()
if __name__ == '__main__':
    while True:
        n_gen = input("Enter number of generations: ")
        if str(n_gen).isdigit():
            n_gen = int(n_gen)
            break
    while True:
        n_iter = input("Enter number of iterations: ")
        if str(n_iter).isdigit():
            n_iter = int(n_iter)
            break
    train(n_iter,n_gen)
    # load_model(main_net, "model2.pt")
    # validate(main_net)
    # print(sys.getsizeof(labels))
