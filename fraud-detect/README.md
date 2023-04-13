# fraud-detect

For this project the data from https://www.kaggle.com/c/ieee-fraud-detection/data is used. (you only have to create a ./data folder inside the main path of the repository and put the data files inside.


# the approach

The approach used is linear:
- First of all the data is pre-processed and cleaned. The data sources are represented by two different files (that need to be joined together), one with the user id information and the other with the information about the transactions.
- Some encoding is required by the moment some variables are qualitative and have string format.
- Once the data is ready, the training is carried out with two algorithms: Decision Tree and XGBoost. 
- For both the algorithms the random search algorithm is used to find the best combination of hyperparameters (This phase only runs once).


