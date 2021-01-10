# # Denoising Autoencoder (DAE) Model

The idea of a de-noise autoencoder is to auto-generate features from the outputs of each hidden layer of the DAE network. The original components were through the noise-swap process as the input of the model. It means each original feature in one training sample has an absolute chance to be swapped its value with another example. The training target is to use those noise swap features to predict the original features. After training the model, outputs from each hidden layer were collected with the original features as inputs. Thus, the results could be the input features for the downstream task, which is the multi-label classification in this case.

The shared model inspired by this discussion https://www.kaggle.com/c/lish-moa/discussion/195642. It has three hidden layers, and the output from each layer went through a weight-normal process before non-linear transformed by a relu activation function. This model was designed to be "wide" first, then to be "narrow", and finally to be wide again.

In training, the swap probability is a special hyperparameter needs to be tuned, besides other general hyperparameters. The "mean square loss" was the error function to be minimized. The model used the Adam gradient descent method and was paired with ReduceLROnPlateau as the training learning rate scheduler. 



Then, those generated features from DAE were used as the input for the Feedforward neural network model. The Adam was selected as the gradient descent method OneCycleLR as the learning rate scheduler. 

## Online inference issue

Due to the competition requirement, results submission had to be processed inside the Kaggle notebook environment. However, the submission than the maximum allowed time limit, so it failed to submission. The submission failure could be the different online feature generation of the private dataset from the DAE model. Due to the DAE model being tested for final submission one day before the competition ended, we had no time to debug the model to submit it with another model. 

## Future Improvement

- Compact the code for final submission to debug. -
