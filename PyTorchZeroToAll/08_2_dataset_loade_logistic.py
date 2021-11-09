# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, from_numpy, optim
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('Learning devices:',device)
if USE_CUDA :
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
#dataset= dataset#.cuda
train_loader = DataLoader(dataset=dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0)


class Model(nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 2)
        self.l4 = nn.Linear(2, 1)
        # self.l5 = nn.Linear(3, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.relu(self.l1(x))
        out2 = self.relu(self.l2(out1))
        out3 = self.relu(self.l3(out2))
        y_pred = self.relu(self.l4(out3))
        
        # y_pred = self.sigmoid(self.l5(out4))

        # y_pred = self.sigmoid(self.l3(out2))
        return y_pred

def plotLoss(e_epoch,l_loss):
    plt.plot(e_epoch,l_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
# our model
model = Model()#.to("device")
# Hyper parameters 
learning_rate = 0.00001
epochs = 1000
# batch_size : data loader

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')
# optimizer = optim.SGD(model.parameters(), lr= learning_rate)
optimizer = optim.Adagrad(model.parameters(), lr= learning_rate)

# Training loop
list_epoch = []
list_loss = []
tries = 0
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tries += 1
    if epoch % 100 == 0:
        prediction = y_pred >= torch.FloatTensor([0.5]) 
        correct_prediction = prediction.float() == labels 
        accuracy = correct_prediction.sum().item() / len(correct_prediction) 
        print('Epoch {:4d}/{} Loss: {} Accuracy {:2.2f}%'.format( 
            epoch, epochs, loss.item(), accuracy * 100,
        ))
        # print(f'Epoch {epoch + 1} | Loss: {loss.item():.4f}')

    list_epoch.append(epoch)
    list_loss.append(loss.item())
plotLoss(list_epoch,list_loss)
