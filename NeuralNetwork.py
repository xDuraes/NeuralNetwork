import numpy as np

class neural_network:
    # TODO: ADD A WAY TO USE DIFFERENT HIDDEN LAYERS ACTIVATION FUNCTION AND LAST LAYER ACTIVATION FUNCIONS (GENERALLY: HIDDEN ARE RELU OR TANH)
    # TODO: MAKE THE FORWARD PROP USE A DIFF LAYER IN THE OUTPUT -> LEN(WEIGHTS)-2 AND OUTPUT ACT FUNC AFTER LOOP


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
        """

        self.architecture = architecture
        self.cache = dict()
        self.learning_rate = learning_rate
        
        # weights initialized using random numbers from normal distribution
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
        
        if task.lower() in ["binary_classification", "multiclass_classification", "regression"]:
            self.task = task.lower()            
            if self.task == "binary_classification":
                self.output_layer_activation_function = activation_functions_available["sigmoid"]
            elif self.task == "multiclass_classification":
                self.output_layer_activation_function = activation_functions_available["softmax"]
            else:
                pass
        else:
            raise NameError("Task not supported")

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
        
    def input_data(self, data, y):
        """
        Setup of the input data in which will occur the training
        """
        # TODO validation
        self.data = data
        self.y = y
        self.num_instances = self.data.shape[1]

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
        # dict to save the values of each layer
        A = self.data
        self.cache["A0"] = A
        for layer_idx in range(len(self.weights)):
            Z = self.weights[layer_idx] @ A + self.biases[layer_idx]
            # different activation function on the output layer
            A = self.output_layer_activation_function(Z) if (layer_idx == (len(self.weights) - 1)) else self.hidden_layers_activation_function(Z)
            self.cache[f"A{layer_idx + 1}"] = A
            self.cache[f"Z{layer_idx + 1}"] = Z
        y_hat = A
        return y_hat
    
    def _calculate_loss(self, y_hat):
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
        """
        
        # original data and prediction
        y = self.y

        if self.task == "binary_classification":
            # BCE loss calculation based on the matrices
            prediction_losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))

        elif self.task == "multiclass_classification":
            # CCE loss calculation based on the matrices
            prediction_losses = -(np.sum((y * np.log(y_hat)), axis=0, keepdims=True))

        # global loss
        losses_sum = (1 / self.num_instances) * np.sum(prediction_losses, axis=1)

        return np.sum(losses_sum)

    def _backpropagation(self, y_hat, y_real):
        """ULTIMO LAYER: DERIVADA DA FUNCAO DE ATIVAÇÃO ISOLADA"""
        gradient_W = [None] * len(self.weights)
        gradient_b = [None] * len(self.weights)

        # last layer dZ for a sigmoid and bce - useful for binary classification
        # only one that uses explicitly the derivative dC/dZ
        # predicted value - real value * scalar factor (1/m)
        if self.task == "binary_classification" or self.task == "multiclass_classification":
            dZ = (y_hat - y_real)

        for layer_idx in reversed(range(len(self.weights))):
            
            # correto (dW = dZ * A^t[l-1])
            A_prev = self.cache[f"A{layer_idx}"]
            dW = ((1/self.num_instances) * (dZ @ A_prev.T))
            
            #correto (db = sum(dZ))
            db = ((1/self.num_instances) * (np.sum(dZ, axis=1, keepdims=True)))

            # saving the weights and biases gradients
            gradient_W[layer_idx] = dW
            gradient_b[layer_idx] = db

            if layer_idx > 0:
                # correto - derivada da camada anterior (w^t * dz)
                dA_back = self.weights[layer_idx]
                dA_back = dA_back.T @ dZ

                dZ = dA_back * self.hidden_layers_derivative_activation_function(self.cache[f"Z{layer_idx}"])

        for layer_idx in range(len(self.weights)):
            # weights and biases adaptation using gradient descent
            # theta = theta - learning rate * slope     (derivative)
            self.weights[layer_idx] -= self.learning_rate * gradient_W[layer_idx]
            self.biases[layer_idx] -= self.learning_rate * gradient_b[layer_idx]

    def train(self, max_iterations:int=10000, min_loss_difference:float=None, file_to_save_weights_and_biases:str=None):

        for _ in range(max_iterations):
            y_hat = self._forward_propagation()
            loss = self._calculate_loss(
                y_hat=y_hat
                )
            print(loss)
            self._backpropagation(
                y_hat=y_hat,
                y_real=self.y,
                )
        return loss
    
    def predict(self, data):
        A = data
        for layer_idx in range(len(self.architecture)-1):
            Z = self.weights[layer_idx] @ A + self.biases[layer_idx]
            A = self.output_layer_activation_function(Z) if (layer_idx == (len(self.weights) - 1)) else self.hidden_layers_activation_function(Z)
        y_hat = A
        return y_hat #(y_hat>0.5).astype(float) can be used for binary classification