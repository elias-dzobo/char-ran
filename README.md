## A Name Categorisation RNN built from scratch 

### Data Loading 
Our data is a directory with subdirectories for each category of names [ Korean, English, Vietnamese ] etc. 
We convert the data in our data directory from unicode to ascii and create a dictonary for each category of names and the ascii names as a list which is the value.
we create helper function to create tensor representations of each letter in the ascii and then create a tensor representation of a name.
we also create a helper function that takes a potential output vector and returns the category of names it belongs to

### Network Creating 
We create our RNN network class with 3 inputs, the size of the input, size of the hidden states, size of the output. 
we understand that the RNN uses a previous output and previous hidden state to compute the current output.
input -> hidden -> output -> softmax for normalisation 
the initial hidden state is randomly generated and the purpose of training is to find hidden state weights that minimise the loss function 

### Training Network
For training, we get a random training samples over an iteration of 10,000 epochs, train the model, calculate our loss, calculate the gradients and shift our training weights to minimise our loss based on our learning rate. 


### Prediction
To predict a name, we pass the name to a function that converts the name to a tensor and feeds it to the RNN to get a output tensor that is converted to a category using the dictionary we created when loading our data. 


### API Endpoint [in development]
- Save model to MLFLOW 
- Save Category dictionary 
- Use FastAPI to create an endpoint that takes a name and returns a category 