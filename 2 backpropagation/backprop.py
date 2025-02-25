

''' now defining class '''

class NeuralNet:
  def __init__(self, learning_rate, num_hidden_units, sweeps = 100, bias = 1): # sweeps = epochs

    
    ''' random values uniform [-0.5, +0.5] '''
    # Initialize with random weights(used random but not on the fly)
    self.num_hidden_units = num_hidden_units
    self.weights_input_hidden = [[0.29929333780219225, -0.17607305797545514],
                            [0.1307652796814418, 0.2900767821427591],
                            [-0.388500430129188, 0.3171168874411031],
                            [0.31446040052849167, 0.11300897689807798]]
    self.weights_hidden_output =[[0.28830645964044244,
                                0.08092364323270795,
                                -0.17585295790429967,
                                -0.2683081275126247]]

    self.bias_input_hidden = [[-0.31123350167553165],
                              [-0.20499766359836868],
                              [0.12088893734663608],
                              [0.21019782406322474]]
    self.bias_hidden_output = [[-0.41236613011277323]]

    self.lr = learning_rate
    self.sweeps = sweeps # also called epochs
    # self.bias = bias


  ''' activation function: we are gonna receive a matrix so need to rely on each value activation'''
  def sigmoid(self, x): # takes weighted sum to calculate the activations
    result = [[0 for _ in range(len(x[0]))] for _ in range(len(x))] # deep copy to avoid modification of original
    for i in range(len(x)):
      for j in range(len(x[0])):
        x[i][j] = 1 / (1 + 2.718281828459045**(-x[i][j])) # math.exp = 2.718281828459045
    return x
 
    # return activated_value

  ''' need derivative for backpropagation '''
  def sigmoid_derivative(self, x):# takes sigmoid actiavtion value
    result = [[0 for _ in range(len(x[0]))] for _ in range(len(x))] # deep copy to avoid modification of original
    for i in range(len(x)):
      for j in range(len(x[0])):
        result[i][j] = x[i][j] * ( 1- x[i][j]) # elementwise operation
    return result


  ''' function to calculate weighted sum and generate the result  and it is used during the forward pass only'''
  def weighted_sum(self, x, w):
    r1 = len(x)
    c1 = len(x[0])
    # c1 = 1 # constant for now
    r2 = len(w)
    c2 = len(w[0])
    if c1 == r2 : # if the matrices are compatible
      r3, c3 = r1, c2 # new row and column
      result = []
      for i in range(r1):
        temp_row = []
        for j in range(c2): # iterating column till c2
          sum = 0
          for k in range(c1): # iterating row till c1
            # print(x[i][k], w[k][j]) # accessing element to be multiplied
            sum += (x[i][k] * w[k][j])
          # print("####")
          temp_row.append(sum) # adding bias to the weighted sum
        result.append(temp_row)
      return result
    else:
      print(f"Matrix dimensions incompatible: {r1}x{c1} and {r2}x{c2}")
      return None


  ''' Adding bias elementwise'''
  def add_bias(self, x, bias):
    result = [[0 for _ in range(len(x[0]))] for _ in range(len(x))]
    for i in range(len(x)):
      for j in range(len(x[0])):
        result[i][j] = x[i][j] + bias[j][0]
    return result

  ''' loss function (loosely MSE). Returns single value '''
  def loss_function(self, y, y_pred):
    loss = 0
    for i in range(len(y)):
      for j in range(len(y[0])):
        loss += (y[i][j] - y_pred[i][j]) ** 2
        # print((y[i][j] - x[i][j]) ** 2)

    return (loss/2)



  ''' fucntion to broadacast ''' # not using
  def broadcast(self, x,y, sign):
    if sign == '*':
      for i in range(len(x)):
        for j in range(len(x[0])):
          x[i][j]*=y[0][j]
      return x
    if sign == '+':
      print(f'x and y: {x, y}')


  ''' Raw error function '''
  def raw_error(self, y_pred, y):
    raw_error = [[0 for _ in range(len(y_pred[0]))] for _ in range(len(y_pred))] # initialization
    # print('in raw error func', y_pred, y)
    for i in range(len(y_pred)):
      for j in range(len(y_pred[0])):
        raw_error[i][j] = y_pred[i][j] - y[i][j]
    # print('raw errror', raw_error, "\ny_pred", y_pred, '\ny', y)
    return raw_error

  def subtract(self, x, y): # not using
    for i in range(len(x)):
      for j in range(len(x[i])):
        x[i][j] += y[i][j]

  ''' to multiply element_wise between vector and scaler'''
  def scaler_multiply(self, multiplier, y):
    # print(f'x and y: {x, y}')
    result = [[0 for _ in range(len(y[0]))] for _ in range(len(y))]
    for i in range(len(y)):
      for j in range(len(y[0])):
        result[i][j] = y[i][j] * multiplier
    return result

  ''' elementwise operations between two matrices'''
  def elementwise_operation(self, matrix1, matrix2, sign):
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
      for j in range(len(matrix1[0])):
        if sign == "+":
          result[i][j] = matrix1[i][j] + matrix2[i][j]
        elif sign == "*":
          result[i][j] = matrix1[i][j] * matrix2[i][j]
    return result


  ''' to calculate the mean delta bias'''
  def mean(self, matrix1):
    sum = 0
    total = 0
    for i in range(len(matrix1)):
      for j in range(len(matrix1[0])):
        sum += matrix1[i][j]
        total += 1
    return [[sum/total]]

  
  ''' transoposing matrix '''
  def transpose(self, x):

    r1 = len(x)
    c1 = len(x[0])
    r2,c2 = c1, r1
    y = [[0 for _ in range(r1)] for _ in range(c1)] # initializing transposed matrix
    for i in range(c1): # iterate over column
      for j in range(r1): # iterate over row
        y[i][j] = x[j][i]
    return y



  ''' Implementing forward pass '''
  def forward_pass(self, X, y): # takes input
    losses = []
    for sweep in range(self.sweeps): # for each epoch
      y_preds = []
      layer_1_activations = []

      for idx in range(len(X)): # iterating over each sample
        # print(f"SWEEP : {sweep}")
        '''Forward Pass'''
        # input to hiddne
        input_transposed = self.transpose([X[idx]])  # Converting to column vector for multiplication with layer1 weights
        layer_1_weighted_sum = self.weighted_sum(self.weights_input_hidden, input_transposed)
        layer_1_output = [[layer_1_weighted_sum[i][0] + self.bias_input_hidden[i][0]] for i in range(len(layer_1_weighted_sum))]
        layer_1_activation = self.sigmoid(layer_1_output)

        # appending for backprop
        layer_1_activations.append(layer_1_activation)
        print('layer_1_activation: ',layer_1_activation) # 1x2
        print('layer_1_activation[0]: ',layer_1_activations[0])

        # hidden to output layer
        layer_2_weighted_sum = self.weighted_sum(self.weights_hidden_output,layer_1_activation )
        print('layer_2_weighted_sum', layer_2_weighted_sum) # 1x1

        layer_2_output = layer_2_output = [[layer_2_weighted_sum[0][0] + self.bias_hidden_output[0][0]]] # adding bias
        print('layer_2_output_added_bias', layer_2_output) # 1x1
        y_pred = self.sigmoid(layer_2_output) # activation
        print('sigmoid_applied_layer_2', y_pred)
        y_preds.append(y_pred[0]) # keep appending the prediction in the bigger list





      print('y_preds: ', y_preds, 'y: ', y)
      y_formatted = [[y[i][0]] for i in range(len(y))] # convert for loss calculation
      # print(y_preds,  y)
      loss = self.loss_function(y_formatted, y_preds)
      losses.append(loss)
      print('loss: ',loss)

      # now print some logs
      if sweep % 500 == 0:
        print(f"Epoch {sweep}, Loss: {loss:.6f}")
      
      
      # Backpropagation
      '''for output layer'''
      # Calculate error at output
      error = self.raw_error(y_preds, y_formatted)

      # Calculate output layer gradient
      y_preds_derivative = []
      for pred in y_preds: # iterative and claulte the derivate of each y_pred
        y_preds_derivative.append(self.sigmoid_derivative([pred])[0])

      # Calculate delta for output layer  i.e. how much error changes wrt output
      output_delta = []
      for i in range(len(error)):
        delta_row = []
        for j in range(len(error[0])):
          delta_row.append(error[i][j] * y_preds_derivative[i][j]) # error * derivative and this will be used in the next part of the equation
        output_delta.append(delta_row)

      # Calculate weight adjustments for output layer 
      delta_weights_hidden_output = [[0 for _ in range(self.num_hidden_units)]]
      for i in range(self.num_hidden_units):
        sum_delta = 0
        for j in range(len(output_delta)):
          sum_delta += output_delta[j][0] * layer_1_activations[j][i][0] # δout⋅a^T
        delta_weights_hidden_output[0][i] = sum_delta


      # Scale weight adjustments by learning rate
      delta_weights_hidden_output = self.scaler_multiply(-self.lr / len(X), delta_weights_hidden_output) # −η.δout⋅a^T

      # Update output layer weights
      self.weights_hidden_output = self.elementwise_operation(self.weights_hidden_output, delta_weights_hidden_output, "+") # actual update

      # Update output layer bias
      delta_bias_hidden_output = [[0]]
      for i in range(len(output_delta)):
        delta_bias_hidden_output[0][0] += output_delta[i][0] # just need output delta
      delta_bias_hidden_output = self.scaler_multiply(-self.lr / len(X), delta_bias_hidden_output) # scale it
      self.bias_hidden_output = self.elementwise_operation(self.bias_hidden_output, delta_bias_hidden_output, "+") # add it
      
      '''for hidden layer'''
      # Backpropagate error to hidden layer
      hidden_deltas = []
      for i in range(len(X)):
        hidden_delta = [[0] for _ in range(self.num_hidden_units)]
        for j in range(self.num_hidden_units):
          hidden_delta[j][0] = output_delta[i][0] * self.weights_hidden_output[0][j] * self.sigmoid_derivative(layer_1_activations[i])[j][0] # δhidden​=g′(h1​)⊙(δout​⋅W2T​) calculating accorind to the equation for the backprop
        hidden_deltas.append(hidden_delta)

      # Calculate weight adjustments for hidden layer
      delta_weights_input_hidden = [[0 for _ in range(2)] for _ in range(self.num_hidden_units)]
      for i in range(self.num_hidden_units):
        for j in range(2):  # 2 input features
          for k in range(len(X)):
            delta_weights_input_hidden[i][j] += hidden_deltas[k][i][0] * X[k][j] # ΔW1​=−η⋅δhiddenT​⋅x (scale later)

      # Scale and update hidden layer weights
      delta_weights_input_hidden = self.scaler_multiply(-self.lr / len(X), delta_weights_input_hidden) # scaling
      self.weights_input_hidden = self.elementwise_operation(self.weights_input_hidden, delta_weights_input_hidden, "+") # updating weihgt

      # Update hidden layer biases
      delta_bias_input_hidden = [[0] for _ in range(self.num_hidden_units)]
      for i in range(self.num_hidden_units):
        for j in range(len(X)):
          delta_bias_input_hidden[i][0] += hidden_deltas[j][i][0]
      delta_bias_input_hidden = self.scaler_multiply(-self.lr / len(X), delta_bias_input_hidden) 
      self.bias_input_hidden = self.elementwise_operation(self.bias_input_hidden, delta_bias_input_hidden, "+")

    return losses

  def predict(self, X):
    predictions = []
    scaled_predictions = []
    for x in X:
      # Input to hidden layer
      input_transposed = self.transpose([x]) # colum vec
      layer_1_weighted_sum = self.weighted_sum(self.weights_input_hidden, input_transposed) # weighted sum
      layer_1_output = [[layer_1_weighted_sum[i][0] + self.bias_input_hidden[i][0]] for i in range(len(layer_1_weighted_sum))] # bias adding
      layer_1_activation = self.sigmoid(layer_1_output) # sig activated

      # Hidden to output layer
      layer_2_weighted_sum = self.weighted_sum(self.weights_hidden_output, layer_1_activation) # previous layer oyutput goes here
      layer_2_output = [[layer_2_weighted_sum[0][0] + self.bias_hidden_output[0][0]]] # bias added 
      y_pred = self.sigmoid(layer_2_output) # output
      
      # manual threshold(other fucntion can be used)
      scaled_prediction = 0
      if y_pred[0][0]>0.5:
        scaled_prediction = 1
      
      scaled_predictions.append(scaled_prediction)
      predictions.append(y_pred[0][0])
    return [predictions, scaled_predictions]


# inputs

X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]
y = [[0],[1],[1],[0]]

nn = NeuralNet(learning_rate=0.8, num_hidden_units=4, sweeps=5000)
losses = nn.forward_pass(X, y)

predictions, scaled_predictions = nn.predict(X)
print("\nFinal Predictions:")
for i in range(len(X)):
  print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i]:.4f}, Scaled_Predictions: {scaled_predictions[i]}")
