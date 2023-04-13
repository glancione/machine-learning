# fraud-detect

For this project the data from https://www.kaggle.com/c/ieee-fraud-detection/data is used. 
Store the data in the ./data folder inside the main path of the repository.


# the approach

The approach is linear:
- First of all the data is pre-processed and cleaned. The data sources are represented by two different files, one with the user id information and the other with the information about the transactions.
- Encoding required, some variables are qualitative and have string format.
- Once the data is ready, the training takes place (Decision Tree and XGBoost). 
- For both the algorithms the random search is used to find the best combination of hyperparameters (This phase only runs once).


