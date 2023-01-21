!unzip /content/netflix.zip

import os
import numpy as np
import pandas as pd

training_data_path = ""
testing_data_path = ""

current_working_directory = os.getcwd()
for file in os.listdir(current_working_directory):
  if "Train" in file:
    training_data_path = current_working_directory + "/" + file
  elif "Test" in file:
    testing_data_path = current_working_directory + "/" + file

training_data = pd.read_csv(training_data_path, header=None, names=["MovieID", "UserID", "Rating"])

training_matrix = pd.pivot_table(training_data, index="UserID", columns="MovieID", values="Rating")
training_matrix = training_matrix.sort_index(axis=0)
normalized_training_matrix = training_matrix.sub(training_matrix.mean(axis=1), axis=0)
normalized_training_matrix = normalized_training_matrix.fillna(0)
normalized_training_matrix = np.array(normalized_training_matrix, dtype = np.float16)

training_matrix_norm = np.linalg.norm(normalized_training_matrix, axis=1)
weights = []
for i,j,k in zip(range(28978), normalized_training_matrix, training_matrix_norm):
     numerator = np.dot(j, normalized_training_matrix.T)
     denominator = k*training_matrix_norm
     denominator = np.where(denominator == 0, 1, denominator)
     weights.append(numerator/denominator)
weights = np.asarray(weights)
np.fill_diagonal(weights, 0)
weights = np.clip(weights, 0, 1000)

k = np.divide(1, (np.sum(np.absolute(weights), axis=1)))
np.nan_to_num(k, 0)

c = pd.unique(training_data.loc[:, "MovieID"])
r = pd.unique(training_data.loc[:, "UserID"])
mean = training_matrix.mean(1)
mean = mean.to_numpy()

predict = np.dot(w, normalized_training_matrix)
predict = np.multiply(k, predict.T).T
predict = np.add(mean, predict.T).T
predict = np.round(predict)

testing_data = pd.read_csv(testing_data_path, header=None, names=["MovieID", "UserID", "Rating"])

test_data = testing_data.to_numpy()
prediction = []
for data in test_data:
    user = np.where(r == data[1])
    movie = np.where(c == data[0])
    predicted = predict[user, movie]
    predicted = predicted[0]
    prediction.append(predicted)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

actual = test_data[:, 2]
rmse = (mean_squared_error(actual, prediction))**0.5
mae = mean_absolute_error(actual, prediction)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error :", mae)