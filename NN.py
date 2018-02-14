import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg



def Predict(Theta1,Theta2,X):
    temp = np.ones((m,1))
    X = np.append(temp,X,axis=1)

    z2 = np.matmul(X,Theta1.T)
    a2 = sigmoid(z2)

    temp = np.ones((a2.shape[0],1))
    a2 = np.append(temp,a2,axis=1)

    z3 = np.matmul(a2,Theta2.T)
    a3 = sigmoid(z3)

    pred = np.argmax(a3, axis=1)+1
    pred = np.expand_dims(pred, axis=1)
    return pred


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out,fan_in+1))
    W = np.sin(np.arange(1,W.size+1))/10
    W = W.reshape((fan_out,fan_in+1))
    return W

def checkGradients(lam=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    theta = np.concatenate([Theta1.ravel(),Theta2.ravel()])
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.remainder(np.arange(1,m+1), num_labels)
    y = np.expand_dims(y,axis=1)
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    e = 0.0001
    for p in range(theta.size):
        perturb[p] = e
        loss1, grad = CostFunction(theta+perturb,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
        loss2, grad = CostFunction(theta-perturb,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
        numgrad[p] = (loss1-loss2)/(2*e)
        perturb[p] = 0
    J,grad = CostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
    return numgrad,grad

# ----------------Random Weights----------------#
def randInitializeWeights(l_in, l_out):
    epsilon_init = 0.12
    W = np.random.rand(l_out, 1+l_in)*2*epsilon_init - epsilon_init
    return W



#-----------------Activation Function------------#

def sigmoid(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j] = 1.0/(1.0+np.exp(-z[i][j]))
    return z

def sigmoidGradient(z):
    g = np.zeros(z.shape)
    a = sigmoid(z)
    g = a*(1-a)
    return g


#----------------Cost Function-------------------#
def CostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam):
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    #print(Theta1.shape,Theta2.shape)

    J = 0
    m = X.shape[0]

    yd = np.identity(num_labels)
    y = yd[y-1,:]
    y = np.squeeze(y,axis=1)

    temp = np.ones((m,1))
    X = np.append(temp,X,axis=1)

    z2 = np.matmul(X,Theta1.T)
    a2 = sigmoid(z2)

    temp = np.ones((a2.shape[0],1))
    a2 = np.append(temp,a2,axis=1)

    z3 = np.matmul(a2,Theta2.T)
    a3 = sigmoid(z3)

    logf = -y*np.log(a3)-(1-y)*np.log(1-a3)
    Theta1s = Theta1[:,1:]
    Theta2s = Theta2[:,1:]

    J = 1.0/m*sum(sum(logf)) + 0.5*lam/m*sum(sum(Theta1s**2)) + 0.5*lam/m*sum(sum(Theta2s**2))
    # grad = np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])
    # grad = np.hstack([Theta1_grad.flatten(),Theta2_grad.flatten()])

    return J

def gradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam):
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    #print(Theta1.shape,Theta2.shape)
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    J = 0
    m = X.shape[0]

    yd = np.identity(num_labels)
    y = yd[y - 1, :]
    y = np.squeeze(y, axis=1)

    temp = np.ones((m, 1))
    X = np.append(temp, X, axis=1)

    z2 = np.matmul(X, Theta1.T)
    a2 = sigmoid(z2)

    temp = np.ones((a2.shape[0], 1))
    a2 = np.append(temp, a2, axis=1)

    z3 = np.matmul(a2, Theta2.T)
    a3 = sigmoid(z3)

    Theta1s = Theta1[:, 1:]
    Theta2s = Theta2[:, 1:]

    a1 = X
    delta3 = np.subtract(a3, y)
    temp = np.ones((m, 1))
    z2 = np.append(temp, z2, axis=1)
    # print(delta3.shape,Theta2.shape)
    delta2 = np.matmul(delta3, Theta2) * sigmoidGradient(z2)
    delta2 = delta2[:, 1:]
    Theta1_grad = Theta1_grad + 1.0 / m * (np.matmul(delta2.T, a1))
    Theta2_grad = Theta2_grad + 1.0 / m * (np.matmul(delta3.T, a2))

    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + lam / m * Theta1s
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + lam / m * Theta2s

    grad = np.hstack([Theta1_grad.flatten(),Theta2_grad.flatten()])
    return grad


#------------------loading Data-------------------#
mat = scipy.io.loadmat('ex4data1.mat')
X = mat['X']
y = mat['y']
m = X.shape[0]

# rand = np.random.randint(5000, size=100)
# sel = X[rand,:]

randx = random.randint(0,4999)
sel = X[randx,:].reshape((20,20))
plt.imshow(sel, cmap = 'gray')
# plt.show()

#---------------loading parameters----------------#

mat = scipy.io.loadmat('ex4weights.mat')
Theta1 = mat['Theta1']
Theta2 = mat['Theta2']

# print("Size of theta1 : "+ str(Theta1.shape))
# print("Size of theta2 : "+ str(Theta2.shape))

#------------ Compute Cost (Feedforward)-----------#
lam = 0
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

nn_params = np.concatenate([Theta1.ravel(),Theta2.ravel()])

# J,grad = CostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
#print(J,grad.shape)



#------------------ Random weights-------------------#

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate([initial_Theta1.ravel(),initial_Theta2.ravel()])


#------------------Gradient Checking----------------#
# numgrad,grad = checkGradients()

lam = 1
#debug_J,debug_grad = CostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lam)
#print(debug_grad)


# ----------------- Training -----------------------#
print("Training....")
theta = fmin_cg(CostFunction, initial_nn_params, fprime = gradient, args=(input_layer_size,hidden_layer_size,num_labels,X, y, lam), maxiter=50)
# print(theta)

Theta1 = theta[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = theta[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)

pred = Predict(Theta1,Theta2,X)

print("Train data Accuracy : "+ str(np.mean((pred == y).astype(int))*100))