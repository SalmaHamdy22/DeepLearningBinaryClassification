
# Heart Disease Prediction with Keras

The goal of this project is to build a binary classification model using Keras to predict heart disease based on various health indicators from the provided dataset.



# Dataset description

The dataset contains 22 columns, with HeartDiseaseorAttack as the target variable. It consists of various health indicators, including:

High Blood Pressure (HighBP)
Body Mass Index (BMI)
Smoking status (Smoker)
Diabetes status (Diabetes)
And other relevant health metrics.



## Steps to Run the Code in Jupyter Notebook

-Load the Dataset:

Import the necessary libraries.
Load the dataset into a Pandas DataFrame.
Split the data into features (X) and target (y).

-Preprocess the Data:

Normalize the features to ensure all inputs to the neural network are on a similar scale.
Split the dataset into training and validating sets (e.g., 80% training, 20% validating).

-Build the Neural Network Model:

Create a sequential model using Keras.
Add layers according to the specifications of the task.

-Compile the Model:

Determine the optimizer (e.g., Adam), loss function (e.g., binary cross-entropy), and evaluation metrics (e.g., accuracy).
Train the Model:

Fit the model to the training data and validate using the test data.
Monitor the loss and accuracy during the training process.
Evaluate the Model:

Calculate and print the following metrics:
Accuracy
Confusion Matrix
Precision, Recall, and F1-Score
ROC-AUC curve for performance evaluation.
Visualize the loss (both training and validation) during the model training process to assess how well the model is learning over time.
Hyperparameter Tuning (Bonus):

Perform hyperparameter tuning (e.g., tuning the number of layers, neurons, and learning rate) using Keras Tuner to optimize the model's performance.
## Dependencies and Installation Instructions
bash

!!pip install torch torchvision numpy pandas matplotlib scikit-learn sklearn metrics kerastuner tuners

## Example Output

After training the neural network, you will obtain metrics that help evaluate the model's performance. The visualization of the loss and accuracy will provide insights into the learning process of the model over time.
