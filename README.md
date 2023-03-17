# Indian defence companies stock price prediction
End to End Flask hosted project on Stock price prediction using LSTM

Namaste,

In this application, I have created a Deep learning model to predict the stock prices of the Top 10 Indian defence companies. I have used LSTM to make the model. The dataset was collected from Yahoo Finance and is uploaded on my Kaggle account at :- https://www.kaggle.com/datasets/venubanaras/nse-defence-stocks-india
This is one of the main projects I have created for putting on my resume, and it took almost 5 days to develop, which mainly involved the training of 10 separate models. and the rest of the time for creating the front end and the back end of the website.

To run the code, simply download this repo and make sure you have all the required libraries installed on your system.
After installing, simply open your terminal or powershell and type :- python app.py
Then open your browser to localhost:5000/home

!! NOTICE !! The /home is necessary as the routing of the website was not my main task and I did what was the easiet to route to the project. 

To create a CSV file using Yahoo Finance, simply go to the website, enter the stock name, select historical data , select the time period (I selected MAX, corresponding to the company's listing date), select historical prices and the frequency (i.e. daily,weekly or monthly) and then download. It will automatically create a .csv file for you to work upon.

The main purpose of this app was to create a deployment of a DL app and the second purpose was to learn Time Series forecasting and also to learn creating datasets for DL projects. I have used TensorFlow to create the models and Flask to create a deployment version for the same. The detailed explanation for the code is in the files itself.
The repo has the full code for development of the model including the CSV files.

To check out my notebook at Kaggle follow :- https://www.kaggle.com/venubanaras
Please like this repo if you enjoyed this project.
Do follow me on Kaggle and give upvotes to the dataset and the notebooks, if you like them.
