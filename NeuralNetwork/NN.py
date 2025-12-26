from constants import SEED
from typing import Optional
import random
import numpy as np
import cv2

class NeuralNetwork:

    def __init__(
            self, 
            architecture:list=[2,3,3,1], 
            hidden_layers_activation_function:str="relu", 
            task:str="binary_classification",
            learning_rate:float=0.01,
            ):
        """
        Initialization parameters of the Neural Network:
            - architecture: a list that contains in each index the number of neurons for each layer 
                            (layer[0] is the number of input neurons and layer[-1] is the number of output neurons)
            - hidden_layers_activation_function: a string with the name of the activation function of each hidden layer neuron
                                                 "sigmoid", "relu" and "tanh" are the activation functions available
            - task: the task in which the neural netword will be utilized, this defines automatically the output layer activation function and the loss function
                    "binary_classification", "multiclass_classification" and "regression" are the tasks available
            - learning_rate: a float that determines each iteration step rate in the gradient descent used in backpropagation

        Stay aware that the number of neurons in the last layer in each task must be exactly as follow
            - binary_classification:  1 neuron, outputs either 1 or 0
            - multiclass_classification: the same number of classes available, outputs the probability for each class and choses the biggest one
            - regression: 1 neuron, outputs any number in R
        """

        assert isinstance(architecture, list), "The architecture of the Neural Network must be a list"
        self.architecture = architecture
        self.cache = dict()
        assert isinstance(learning_rate, float), "The learning rate must be a float"
        self.learning_rate = learning_rate
        
        # weights initialized using random numbers from normal distribution
        # TODO: SEARCH FOR A BETTER WAY TO INITIALIZE THE WEIGHTS
        if hidden_layers_activation_function == "relu":
            self.weights = [
                np.random.randn(architecture[x+1], architecture[x]) * np.sqrt(2 / architecture[x])
                for x in range(len(architecture)-1)
                ]
        else: 
            self.weights = [
                np.random.randn(architecture[x+1], architecture[x]) * np.sqrt(1 / architecture[x])
                for x in range(len(architecture)-1)
                ]
        
        # biases initialized using zeros for each neuron
        self.biases = [
            np.zeros((architecture[x+1], 1))
            for x in range(len(architecture)-1)
        ]

        # activation functions supported
        activation_functions_available = {
            "relu": lambda x: np.maximum(0, x),
            "tanh": np.tanh,
            "sigmoid": lambda x: ((1) / (1 + np.exp(-x))),
            "softmax": lambda x: (np.exp(x - np.max(x, axis=0, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=0, keepdims=True)), axis=0, keepdims=True))
        }
        
        # setup of the output layer activation function based on each specific task
        if task.lower() in ["binary_classification", "multiclass_classification", "regression"]:
            self.task = task.lower()            
            if self.task == "binary_classification":
                self.output_layer_activation_function = activation_functions_available["sigmoid"]
            elif self.task == "multiclass_classification":
                self.output_layer_activation_function = activation_functions_available["softmax"]
            elif self.task == "regression":
                # outputs any R number
                self.output_layer_activation_function = lambda x: x
        else:
            raise NameError("Task not supported")

        # setup of the hidden layers activation function
        if hidden_layers_activation_function.lower() in activation_functions_available.keys():
            self.hidden_layers_activation_function = activation_functions_available[hidden_layers_activation_function.lower()]
        else:
            raise NameError("Activation function not supported")

        derivate_activation_function = {
            # relu'(Z) = 1 if Z > 0 else 0
            "relu": lambda Z: (Z > 0).astype(float),
            # tanh'(Z) = 1 - tanh(Z)²
            "tanh": lambda Z: 1 - np.tanh(Z)**2,
            # sig'(Z) = sig(Z) * (1 - sig(Z))
            "sigmoid": lambda Z: ((1) / (1 + np.exp(-Z))) * (1 - ((1) / (1 + np.exp(-Z))))
        }

        self.hidden_layers_derivative_activation_function = derivate_activation_function[hidden_layers_activation_function.lower()]
        
    def input_data(self, data:list, data_y:list, split_train:float=0.85, is_image:bool=False, preserve_rgb:bool=False):
        """
        Setup of the input data in which will occur the training

        Parameters:
            - data: a list that contains the data that will be used for training and testing
                1. Tabular data: the list must contain a lists with the features of each instance
                1. Image data: the list must contain the path to each image
            - data_y: a list that contain the target feature of each instance, 
                1. Binary Classification: all target features must be either 1 or 0
                1. Multiclass Classification: all target features must be a str with the name of each class
                1. Regression: all target features must be floats
            - split_train: the percentage of the dataset that will be used for training, the rest will be used for testing
            - is_image: a bool that determines if the input data will be images
            - preserve_rgb: a bool that determines if the image will be in the RGB color scheme or in grayscale
                1. If True: 
                    - The training will have heavier computational cost
                    - The training will have more time dedicated to it
                    - The imput layer will have to be 3x the number of pixels
        """
# ======================
# TODO: CHECK IF I HAVE TO NORMALIZE THE INPUTS FOR REGRESSION
# ======================
        def _get_split(X:np.array, Y:np.array, total_instances:int, split_train:float):
            """
            Split of the data into train and test data
            """
            split = int(total_instances * split_train)

            X_train = X[:, :split]
            X_test  = X[:, split:]
            
            Y_train = Y[:split]
            Y_test  = Y[split:]

            return X_train, X_test, Y_train, Y_test
        
        def _set_multiclass_y_data(data_Y:np.array, num_classes:int):
            """
            One-Hot-Encoding of the target variable for multiclass classification
            """
            Y = np.zeros((num_classes, len(data_Y)))
            for i, label in enumerate(data_Y):
                Y[self.label2id[label], i] = 1
            return Y
        
        # shuffling the input data
        random.seed(SEED)
        zip_data = list(zip(data, data_y))
        random.shuffle(zip_data)
        data, data_y = zip(*zip_data)
        data = list(data)
        data_y = list(data_y)

        self.is_image = is_image

        num_classes = self.architecture[-1]

        # in regression we don't use classes, so no need to store in memory these dicts
        if self.task != "regression":
            classes = sorted(set(data_y))

            self.label2id = {label: idx for idx, label in enumerate(classes)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # normalization of each input data according to it's type
        if is_image:
            self.preserve_rgb = preserve_rgb

            data = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB if preserve_rgb else cv2.COLOR_BGR2GRAY) for img_path in data]
            # reshaping each image and normalizating it according to the range available for each pixel
            data = [img.reshape(-1) for img in data]
            data = np.array(data) / 255.0
            X = data.T
            
            total_num_instances = X.shape[1]

            # splitting the data
            self.X_train, self.X_test, Y_train, Y_test = _get_split(
                X=X, 
                Y=data_y, 
                total_instances=total_num_instances, 
                split_train=split_train
            )

            # setup of the target feature 
            if self.task == "multiclass_classification":
                self.Y_train = _set_multiclass_y_data(
                    data_Y=Y_train,
                    num_classes=num_classes
                )
                self.Y_test = _set_multiclass_y_data(
                    data_Y=Y_test,
                    num_classes=num_classes
                )
            elif self.task == "binary_classification":
                # setting the array to a shape that the network accepts
                self.Y_train = np.array(Y_train).reshape(1, -1)
                self.Y_test = np.array(Y_test).reshape(1, -1)

            self.num_instances = self.X_train.shape[1]
        
        else:
            data = np.array(data)

            X = data.T

            total_num_instances = X.shape[1]

            # splitting the data
            X_train, X_test, Y_train, Y_test = _get_split(
                X=X, 
                Y=data_y, 
                total_instances=total_num_instances, 
                split_train=split_train)

            # setup of the target feature 
            if self.task == "multiclass_classification":
                self.Y_train = _set_multiclass_y_data(
                    data_Y=Y_train,
                    num_classes=num_classes
                )
                self.Y_test = _set_multiclass_y_data(
                    data_Y=Y_test,
                    num_classes=num_classes
                )
            # regression available only for tabular data
            elif self.task in ["binary_classification", "regression"]:
                # setting the array to a shape that the network works with  
                self.Y_train = np.array(Y_train).reshape(1, -1)
                self.Y_test = np.array(Y_test).reshape(1, -1)

            self.num_instances = self.X_train.shape[1]

            # tabular data normalized using the z-score normalization
            # Z = ((X - mean) / std)
            # this statistical metrics based on the training data will be used to normalize the prediction data aswell
            self.X_train_mean = X_train.mean(axis=1, keepdims=True)
            self.X_train_std = X_train.std(axis=1, keepdims=True)

            # normaling all training data according to the previous statistical metrics
            self.X_train = ((X_train - self.X_train_mean) / self.X_train_std)
            self.X_test = ((X_test - self.X_train_mean) / self.X_train_std)

    def _forward_propagation(self):
        """
        Basic formula:
        Z[l] = W[l] A[l-1] + b[l]
        A[l] = g(Z[l])
        
        Where:
            - l: Current Layer
            - W: Weights
            - A: Activation Vector
            - b: Biases
            - g: Activation Function
        """

        # input data matrix
        A = self.X_train
        # dict to save the values of each layer
        self.cache["A0"] = A
        for layer_idx in range(len(self.weights)):
            Z = self.weights[layer_idx] @ A + self.biases[layer_idx]
            # different activation function on the output layer
            A = self.output_layer_activation_function(Z) if (layer_idx == (len(self.weights) - 1)) else self.hidden_layers_activation_function(Z)
            # caching the values to use them in the backpropagation
            self.cache[f"A{layer_idx + 1}"] = A
            self.cache[f"Z{layer_idx + 1}"] = Z
        y_hat = A
        return y_hat
    
    def _calculate_loss(self, y_hat, y_real):
        """
        Binary Cross Entropy (BCE) to calculate the loss for binary classification

        For a single example:
            - L(y_hat, y) = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))

        For all training samples:
            - C = (1 / m) * sum(L(y_hat, y))

        Categorial Cross Entropy (CCE) to calculate the loss for multiclass classification

        For a single example:
            - L(y_hat, y) = -(sum(y * log(y_hat)))

        For all training samples:
            - C = (1 / m) * sum(-sum(y log(y_hat)))

        Mean Squared Error (MSE) to calculate the loss for regression

        For a single example:
            - L(y_hat, y) = ((y_real - y_pred) ** 2)

        For all training samples:
            - C = (1 / m) * sum((y_real - y_pred) ** 2)
        """
        
        if self.task == "binary_classification":
            # BCE loss calculation based on the matrices
            prediction_losses = -((y_real * np.log(y_hat)) + (1 - y_real) * np.log(1 - y_hat))

        elif self.task == "multiclass_classification":
            # CCE loss calculation based on the matrices
            prediction_losses = -(np.sum((y_real * np.log(y_hat)), axis=0, keepdims=True))

        elif self.task == "regression":
            # MSE loss calculation based on the matrices
            prediction_losses = ((y_real - y_hat) ** 2)

        # global loss
        losses_sum = (1 / self.num_instances) * np.sum(prediction_losses, axis=1)

        return np.sum(losses_sum)

    def _backpropagation(self, y_hat, y_real):
        """
        Backpropagation based on the derivatives of each layer using the derivative chain rule propagating the derivatives 
        to the previous layer to calculate its slope and adapt the weights and biases values with gradient descent to advance into the global minimum
        of the loss value by going into the opossite direction of the slope

        Gradient Descent:
            - Step Size = slope * learning rate
            - Theta = Theta - step size
        """

        # dicts to save the weights and biases gradients to modify the
        gradient_W = [None] * len(self.weights)
        gradient_b = [None] * len(self.weights)

        # last layer derivative (dZ) 
        # only one that uses explicitly the derivative dC/dZ
        # dZ = predicted value - real value
        if self.task in ["binary_classification", "multiclass_classification", "regression"]:
            dZ = (y_hat - y_real)

        for layer_idx in reversed(range(len(self.weights))):
            
            # dW = dZ * A^t[l-1]
            A_prev = self.cache[f"A{layer_idx}"]
            dW = ((1/self.num_instances) * (dZ @ A_prev.T))
            
            # db = sum(dZ)
            db = ((1/self.num_instances) * (np.sum(dZ, axis=1, keepdims=True)))

            # saving the weights and biases gradients
            gradient_W[layer_idx] = dW
            gradient_b[layer_idx] = db

            if layer_idx > 0:
                # previous layer derivative
                # w^t * dz
                dA_back = self.weights[layer_idx]
                dA_back = dA_back.T @ dZ

                dZ = dA_back * self.hidden_layers_derivative_activation_function(self.cache[f"Z{layer_idx}"])

        for layer_idx in range(len(self.weights)):
            # weights and biases adaptation using gradient descent
            # theta = theta - learning rate * slope (derivative)
            self.weights[layer_idx] -= self.learning_rate * gradient_W[layer_idx]
            self.biases[layer_idx] -= self.learning_rate * gradient_b[layer_idx]

    def train(self, max_iterations:int=10000, min_loss_difference:Optional[float]=None, file_to_save_weights_and_biases:Optional[str]=None):
        """
        Three step training:
            1. Forward Propagation to predict a value
            1. Global Loss Calculation to determinate if an early stop will be used
            1. Backward Propagation to adapt the weights and biases using gradient descent

        Initialization Parameters:
            - max_iterations: an int that determines the maximum number of iterations that the process will have
            - min_loss_difference: a float that determines the minimum difference in the loss in two sequential iterations,
                                   this value is optional and determines if an early stop will be allowed to happen
            - file_to_save_weights_and_biases: a str that contains the path to a file to save the weights and biases
                                               achieved in the end of training, this value is options and if it is not
                                               given, the values will not be saved
        """
        # TODO: add a way to save the weights and biases of a trained nn -> creates the necessity to add a boot param to it into the __init__
        #       add an early stop to a min_loss_difference to use when the gradient descent is taking small steps
        for _ in range(max_iterations):
            y_hat = self._forward_propagation()
            loss = self._calculate_loss(
                y_hat=y_hat,
                y_real=self.Y_train
                )
            print(loss)
            self._backpropagation(
                y_hat=y_hat,
                y_real=self.Y_train,
                )
        return loss
    
    def evaluate(self):
        """
        Evaluation of the trained model with the test split separated in the function input_data
        """
        # TODO: add metrics to the eval and train datasets, at least f1, acc, prec, rec
        # try to plot using matplotlib 
        A = self.X_test
        for layer_idx in range(len(self.weights)):
            Z = self.weights[layer_idx] @ A + self.biases[layer_idx]
            # different activation function on the output layer
            A = self.output_layer_activation_function(Z) if (layer_idx == (len(self.weights) - 1)) else self.hidden_layers_activation_function(Z)
        y_hat = A
        loss = self._calculate_loss(y_hat=y_hat, y_real=self.Y_test)
        print(f"Evaluation dataset loss: {loss}")
    
    def predict(self, data):
        """
        Function that predicts a given data based on the previously set model configuration and 
        the weights and biases achieved in the training process. The input data will be processed and
        normalized in the same way that the training data was.

        Parameters:
            - data: the data that will be used for training and testing
                1. Must be structured in the same way that the training data was
        """

        # processing the input predict data to stay in the necessary shape and normalization
        # the normalization and preprocessing data depends on the configurations set in the whole training process
        if self.task in ["multiclass_classification", "binary_classification"]:
            if self.is_image:
                assert isinstance(data, str), "Insert the path to the raw image"
                data = cv2.cvtColor(cv2.imread(data), cv2.COLOR_BGR2RGB if self.preserve_rgb else cv2.COLOR_BGR2GRAY)
                # applying the same normalization used in the train data
                data = data.reshape(-1) / 255.0
                data = data.reshape(-1, 1)
            else:
                # applying the z-score normalization with the statistics obtained in the train data
                data = ((data - self.X_train_mean) / self.X_train_std)
                data = data.reshape(-1, 1)

        # forward propagation to make the prediction
        A = data
        for layer_idx in range(len(self.architecture)-1):
            Z = self.weights[layer_idx] @ A + self.biases[layer_idx]
            A = self.output_layer_activation_function(Z) if (layer_idx == (len(self.weights) - 1)) else self.hidden_layers_activation_function(Z)
        y_hat = A
        
        # standartizing the output based on the task
        if self.task == "multiclass_classification":
            # returns the name of the class that had the biggest probability based on the id2label dict
            idx_class = np.argmax(y_hat, axis=0)[0]
            prediction = self.id2label[idx_class]
        elif self.task == "binary_classification":
            # returns 1 if prediction is bigger than 0.5 else 0
            prediction = int(y_hat > 0.5)
        elif self.task == "regression":
            # returns the value predicted
            prediction = float(y_hat[0,0])

        return prediction