# Stock Price Prediction with LSTM

In today's era, financial forecasts have gained significant popularity. Numerous models are being utilized to develop prediction systems for various assets such as stocks, currencies, and cryptocurrencies. The primary objective is to make informed decisions, whether to buy or sell, in order to maximize profits or asset value. Decision support systems are now being built using the capabilities of machine learning to generate accurate predictions for future financial outcomes.

For this repository, 3 big companies' stock prices, Apple, Samsung, and Xiaomi are used to predict their next 7 days stock prices. The LSTM algorithm is used for prediction. So, 3 different algorithms are created for each company. 


# Data Preparation
The datasets used in this study were sourced from "Investing.com - Stock Market Quotes & Financial News," a platform that aggregates stock price data from various global sources for technology companies. These datasets included historical records at different time intervals, such as hourly, daily, weekly, and monthly. Specifically, stock price data for Apple, Samsung, and Xiaomi were obtained from this platform.

During the data collection process, instances with missing values were identified. These gaps were mainly a result of formal holidays or weekends when stock prices remained unchanged. Additionally, other missing values, including extreme outliers, were detected and treated as outliers, subsequently removed from the dataset. To address the missing values, a Moving Average technique based on the last 10 days' stock price changes was applied. And, the weekends were excluded from the datasets

The latest versions of these datasets have been labeled as V2 and are stored in the Weight&Bias platform's artifacts. Each notebook is available below the Data preparation folder.


# Model Selection
LSTM, which stands for Long Short-Term Memory, is a type of artificial neural network extensively applied in the domains of artificial intelligence and deep learning. What sets LSTM apart is its ability to incorporate long-term memory in a highly efficient manner, allowing it to learn an even greater number of parameters. While LSTM belongs to the category of Recurrent Neural Networks (RNNs), it stands out as the most potent RNN for the task of forecasting, especially when dealing with prolonged trends in data. Deep neural network structures, like LSTM, possess the capability to uncover concealed patterns and dynamics within data, making them well-suited for predictive tasks. Consequently, LSTM was chosen as the primary model for this study, and all development work was carried out using the LSTM architecture.


# Model and Experiments

The datasets were split as 80% of the dataset was for training, 10% of the dataset was for validation, and 10% of the dataset was for testing. The standard scaling was applied for the split datasets. For only the Apple dataset, the dataset had to be filtered to get high performance on the model prediction, and the data after 2020 was used. This filtering process was not applied to the other datasets belonging to Samsung and Xiaomi.

## Baseline Model
As a first step of the model development process, a baseline model, Naive forecast was created and the Naive forecast results were obtained to evaluate the model performances for each company. The naive forecasting method simply states that the current period will be the same as the previous period. Naïve Prediction is a simple prediction and assumption that the previous day’s values are valid for the following day. So the Naive forecast MAPE test results; Apple: 1.66%, Samsung: 1.1%, Xiaomi: 2.28% were obtained. You can find the notebook below the 'Baseline' folder. This notebook can be applied for all datasets according to the keywords of companies.

## Architecture Tuning
As a second step, the architectures of the LSTM models were created by Bayesian hyperparameter tuning which was applied only to determine the number of hidden layers and nodes. This tuning gave the optimal number of nodes and hidden layers for each company. These values of the number of nodes and layers were kept as the architecture of the LSTM models. You can find this process below the  'Hyperparameter Tuning' folder in the architecture-tuning.ipynb.

## Hyperparameter Tuning

As a third step, after the architecture tuning, the following steps were applied to the LSTM model for the hyperparameter tuning and model performance improvement;
* Xavier initialization 
* Learning rate scheduler 
* Batch size tuning 
* Sequence size
By this hyperparameter tuning, the optimal model for each company was saved. The optimal models were detected with respect to the determined threshold. You can find this process in the 'fine-tuning.ipynb' notebook.

# Results
![MAPE-Baseline-LSTM](/Graphs/MAPE-Baseline-LSTM.png)


# Dataset

# Environment

# Notebooks 

# Proposed Resources





# Contribution
If you want to contribute please, send your pull request. All contributions are welcome!

#
Please check that repository for updates, for opening issues or sending pull requests.