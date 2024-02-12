import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as ds
import pandas as pd
from IPython.display import clear_output
import sklearn.preprocessing as pp

df = pd.read_csv('train_dataset.csv')
#normalise data
scaler = pp.StandardScaler()
df[['Elevation (meters)', ' Horizontal_Distance_To_Hydrology (meters)']] = scaler.fit_transform(df[['Elevation (meters)', ' Horizontal_Distance_To_Hydrology (meters)']])


from sklearn.model_selection import train_test_split
#split df into train and test set
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)


def z_x(x, bias, weights):
    """ param x: vector containing measurements. x = [x1, x2,.. x_n]
        param bias: single value
        param weight: vector containing model weights. weights= [w1,w2,.. w_n]
        
        return: value of logistic regression model for defined x, bias and weights
    """
    power = bias
    for i in range(len(x)):
        power += weights[i]*x[i]
    denom = 1 + pow(np.e, -power)
    return 1/denom


def cost_function(y, x, bias, weights):
    """ param y: Ground truth label for measurements
        param x: vector containing measurements. x = [x1, x2,.. x_n]
        param bias: single value
        param weight: vector containing model weights. weights= [w1,w2,.. w_n]
    
        return: value of the cost function. In this case BCE
    """
    sum = 0
    for i in range(len(x)):
        sum += (y*np.log(z_x(x[i], bias, weights)) + (1-y)*np.log(1-z_x(x[i], bias, weights))) #TODO: /m?
    return -sum/len(x)

def derivative_bias(y, x, bias, weights):
    """ param y: Ground truth label for measurements
        param x: vector containing measurements.
        param bias: single value
        param weight: vector containing model weights. weights= [w1,w2]
    
        return: derivative of cost function with respect to the bias
    """
    db = 0
    for i in range(len(x)):
        db += (z_x(x[i], bias, weights) - y[i]) / len(x)
    return db


def derivative_weights(y, x, bias, weights):
    """ param y: Ground truth label for measurements
        param x: vector containing measurements. x = [x1, x2,.. x_n]
        param bias: single value
        param weight: vector containing model weights. weights= [w1,w2,.. w_n]
    
        return: derivative of cost function with respect to the weights, dw = [dw1, dw2]
    """
    dw = [0]*len(weights)
    for i in range(len(x)):
        for j in range(len(weights)):
            dw[j] += (z_x(x[i], bias, weights) - y[i])*x[i][j] / len(x)
    return dw




lr = 0.01 # <-- specify learning rate

y = train_set[' Forest Cover Type Classes']
x = train_set[['Elevation (meters)', ' Horizontal_Distance_To_Hydrology (meters)']]

# Initialize weights and bias as random
bias = np.random.normal()

weights = np.random.normal(size = len(x.columns))


number_of_iterations = 1000 # <-- number of iterations to perform gradient descent

# Loop through training data and update the weights at each iteration

for it in range(number_of_iterations):
    cost = cost_function(y, x, bias, weights)
    
    clear_output(wait=True) # This is used to clear the output for cleaner printing, can be removed if it causes trouble.
    print('iteration: ', it, ' cost: ', cost) # In this case the variable for the current cost is called "cost"
    dw = derivative_weights(y, x, bias, weights)
    for i in range(len(weights)):
        weights[i] -= lr * dw[i]

    bias -= lr * derivative_bias(y, x, bias, weights) 