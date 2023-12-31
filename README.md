# Stock Price Prediction with LSTM

In today's era, financial forecasts have gained significant popularity. Numerous models are being utilized to develop prediction systems for various assets such as stocks, currencies, and cryptocurrencies. The primary objective is to make informed decisions, whether to buy or sell, in order to maximize profits or asset value. Decision support systems are now being built using the capabilities of machine learning to generate accurate predictions for future financial outcomes.

For this repository, 3 big companies' stock prices, Apple, Samsung, and Xiaomi are used to predict their next 7 days stock prices. The LSTM algorithm is used for prediction. So, 3 different algorithms are created for each company. 


## Dataset and Preprocessing
The datasets used in this study were sourced from [Investing.com][dataset-link] a platform that aggregates stock price data from various global sources for technology companies. These datasets included historical records at different time intervals, such as hourly, daily, weekly, and monthly. Specifically, stock price data for Apple, Samsung, and Xiaomi were obtained from this platform.

During the data collection process, instances with missing values were identified. These gaps were mainly a result of formal holidays or weekends when stock prices remained unchanged. Additionally, other missing values, including extreme outliers, were detected and treated as outliers, subsequently removed from the dataset. To address the missing values, a Moving Average technique based on the last 10 days' stock price changes was applied. And, the weekends were excluded from the datasets

The latest versions of these datasets have been labeled as V2 and are stored in the Weight&Bias platform's artifacts. Notebooks for companies are available below the [Data Preparation][data-prep] folder.

[dataset-link]: https://www.investing.com/
[data-prep]: https://github.com/ftmoztl/stock-price-prediction-with-LSTM/tree/main/Data%20Preparation


## Model Selection
LSTM, which stands for Long Short-Term Memory, is a type of artificial neural network extensively applied in the domains of artificial intelligence and deep learning. What sets LSTM apart is its ability to incorporate long-term memory in a highly efficient manner, allowing it to learn an even greater number of parameters. While LSTM belongs to the category of Recurrent Neural Networks (RNNs), it stands out as the most potent RNN for the task of forecasting, especially when dealing with prolonged trends in data. Deep neural network structures, like LSTM, possess the capability to uncover concealed patterns and dynamics within data, making them well-suited for predictive tasks. Consequently, LSTM was chosen as the primary model for this study, and all development work was carried out using the LSTM architecture.

## Model and Experiments
The datasets were split as 80% of the dataset was for training, 10% of the dataset was for validation, and 10% of the dataset was for testing. The standard scaling was applied for the split datasets. For only the Apple dataset, the dataset had to be filtered to get high performance on the model prediction, and the data after 2020 was used. This filtering process was not applied to the other datasets belonging to Samsung and Xiaomi.

### Baseline Model
As a first step of the model development process, a baseline model, Naive forecast was created and the Naive forecast results were obtained to evaluate the model performances for each company. The naive forecasting method simply states that the current period will be the same as the previous period. Naïve Prediction is a simple prediction and assumption that the previous day’s values are valid for the following day. So the Naive forecast MAPE test results; Apple: 1.66%, Samsung: 1.1%, Xiaomi: 2.28% were obtained. You can find the notebook [here][baseline]. This notebook can be applied for all datasets according to the keywords of companies.

[baseline]: https://github.com/ftmoztl/stock-price-prediction-with-LSTM/blob/main/Baseline/baseline-models.ipynb

### Architecture Tuning
As a second step, the architectures of the LSTM models were created by Bayesian hyperparameter tuning which was applied only to determine the number of hidden layers and nodes. This tuning gave the optimal number of nodes and hidden layers for each company. These values of the number of nodes and layers were kept as the architecture of the LSTM models. You can find this process below the  'Hyperparameter Tuning' folder in the [architecture-tuning.ipynb][arc].

[arc]: https://github.com/ftmoztl/stock-price-prediction-with-LSTM/blob/main/Hyperparameter%20Tuning/architecture-tuning.ipynb

### Hyperparameter Tuning
As a third step, after the architecture tuning, the following steps were applied to the LSTM model for the hyperparameter tuning and model performance improvement;
* Xavier initialization 
* Learning rate scheduler 
* Batch size tuning 
* Sequence size
By this hyperparameter tuning, the optimal model for each company was saved. The optimal models were detected with respect to the determined threshold. You can find this process in the [fine-tuning.ipynb][fine] notebook.

[fine]:https://github.com/ftmoztl/stock-price-prediction-with-LSTM/blob/main/Hyperparameter%20Tuning/fine-tuning.ipynb

## Results
Threshold values of model performance are determined to stop the parameter tuning process. The thresholds were calculated according to the baseline results (multiplied by 0.1) of each company, and optimal models were saved when the models that have reached this target value. You can find the MAPE results of the baseline and developed model which belongs to each company.

![MAPE-Baseline-LSTM](/Graphs/MAPE-Baseline-LSTM.png)

After developing and saving the model, you can use the saved model to visualize the prediction and compare the prediction and actual values of the stock prices. You can use [get-model-predictions.ipynb][pred] notebooks for this purpose. You can find the sample of real Apple stock prices and predictions with the developed LSTM model below. 

![Apple-Prediction-Actual](/Graphs/Apple-Prediction-Actual.png)

As you can see above, the model obtained a very close prediction of the actual values.

[pred]:https://github.com/ftmoztl/stock-price-prediction-with-LSTM/blob/main/Model%20Predictions/get-model-predictions.ipynb

## Environment
To install the dependencies to run the notebook, you can use Anaconda. Once you have installed Anaconda, run:

`$ conda env create -f environment.yml`

## Useful Notes
- For this study, in the model development process, the W&B platform was used to track, log, and observe the developing process. Especially, for the deep neural network models, to track each parameter's effect, train, and validation loss and important details this platform is very proposed. 

- Torch architecture was preferred for applying LSTM which can make code easier to read and debug. This is especially valuable when working on complex models like LSTMs.

- The model development process was divided into 2 phases. Firstly, architecture tuning is applied to decide the algorithm's essential components,  the number of hidden layers and nodes. After this parameters are determined,  the other parameters should be tuned. This approach is proposed by professors in this field.  


## Contribution
If you want to contribute please, send your pull request. All contributions are welcome!

#
Please check that repository for updates, for opening issues or sending pull requests.