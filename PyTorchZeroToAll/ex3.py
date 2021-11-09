#the loss function y-y_pred = (x^2*w_2 + x*w_1 + b - y)^2
# derivative of w_1 = 2x(xw1+w2x^2+b)
# derivative of w_2 = 2x^2(x^2w2+w1x+b)


import torch
import matplotlib.pyplot as plt


w1 = 1.0
w2= 1.0
b=0
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(x):
    return (x**2)*w2 + x*w1 + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient_w1(x,y):
    return 2*x*(x*w1+w2*(x*x*2)*w2+b)

def gradient_w2(x,y):
    return (x*x*2)*(w2*(x*x)+(w1*x)+b)

def plotLoss(e_epoch,l_loss):
    plt.plot(e_epoch,l_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

#Test for number 4
print("Prediction for number 4 before training", forward(4))
learning_rate=0.00001
train_epoch = 300
ll_loss =[]
e_epoch=[]

for epoch in range(train_epoch):
    for x,y in zip(x_data,y_data):   
        grad_w1= gradient_w1(x,y)
        grad_w2= gradient_w2(x,y)
        w1= w1 - learning_rate*grad_w1
        w2= w2 - learning_rate*grad_w2
        
        #print("\t Gradient %.2f"  %x, "%.2f" %y, "%.2f" %grad)
        l_loss= loss(x,y)
        last_loss = l_loss
    e_epoch.append(epoch)
    ll_loss.append(l_loss)
    
    print("Epoc: ", epoch, " w1 = %.2f" %w1, " w2 = %.2f" %w2,  " loss = %.2f" %(l_loss))   
   
print("Prediction for number 4 after training", forward(4))  
print("%.35f"%min(ll_loss))
plotLoss(e_epoch,ll_loss)