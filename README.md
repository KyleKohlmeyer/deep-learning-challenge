# deep-learning-challenge

## Overview: 

The goal of the analysis is to design a machine learning algorithm to predict whether or not charity funding is used effectively based on the details of the campaign. A neutral network was chosen to accomplish this goal because of how robust of a method it is and because there is not a large need for the model to be very intrepretable.

## Data Preprocessing:

### Targets and features
The target is the IS_SUCCESSFUL value, with a 1 indicating successful use of funds, and a 0 an unsuccessful or misuse of funds. The features are everything else in the dataframe that can be used to predict the outcome. The EIN is neither a useful feature nor is it a measure, so it was dropped from the dataframe and subsequently the analysis. The company names were originally dropped to avoid overfitting, but these were ultimately retained to obtain a more accurate model. If the future data is expected to have a lot of companies that have multiple or frequent campaigns, this makes sense to include as the companies past track record is a good predictor of frequent success. However, if the data the model will be used on mainly contains not previously seen companies, the results will likely suffer by including names in the training. This is because the neural network will be optimized on the companies from the training dataset and will not generalize as well to  new data that contains a lot of companies it hasn't seen before (an overfit model). 

Some of the columns had catagories that held only had a small percentage of the data, particularly some catagories in the "Name", "Application Type", and "Classification" columns. These catagories were all grouped into an "Other" catagory that contained a sufficient value_count to be representative of a significant percentage of the data. Dummy data was obtained for any non-numerical columns, separate dataframes were created for the target value and the measures, and the data was split into training and testing sets. 

## Compiling, Training, and Evaluating the model:

### Optimization
Originally, the neural network contained two hidden layers, each with 25 neurons and a ReLu activation function, and an output layer with a sigmoidal activation function. The model was set to run with 100 epochs. This model was unsuccessful in achieving the desired accuracy, so different combinations of activation functions (including the Tanh activation function) were tried, a 3rd hidden layer was added, and the binning was adjusted. This also failed to achieve the desired results, so the amount of epochs was increased to 2000 and 100 neurons were added to each hidden layer. The increase in neurons and amount of epochs did not change much, as the accuracy on the training data would reach approximately 74.5% around 150 epochs and remain there despite there being hundreds more epochs, and the accuracy on the testing data remained at 72.5%. After consultation with instructor Sakib, I decided to implement a pyramid structure to my neural network, with the first hidden layer having the most neurons and subsequent layers having progressively less. This still did not achieve an accuracy greater than 75%, which prompted the reinclusion of the "Name" data as a measure. 

### Results
After running the pyramidal model with 500 epochs on the updated data, an accuracy of 78.8% was achieved on the testing data, which surpassed the target accuracy of 75%. However, as mentioned above, this accuracy is likely artificially high as the model is overfit to the training and testing data, due to there being a lot of the same companies in both the training and testing datasets that would not appear in new data. 78.8% accuracy even with a potentially overfit dataset is well below what should be the desired target for something as important as ensuring charity money is used effectively, and thus I do not believe this model should be used in its current state. I would recommend either collecting more rows of data if possible, or by collecting additional columns of data for each campaign to add more features to the model.

### Other models:

It would be possible (and perhaps even beneficial) to use other supervised machine learning models such as a logistic regression or random forest, due to the data being in tabular form after the preprocessing steps. The advantage to using one of these models instead of a neural network include better intrepretability and using less compute power. To implement one of these models, the only difference would be to set up the chosen model instead of the neural network, as the preprocessing would be the same for any supervised model.
