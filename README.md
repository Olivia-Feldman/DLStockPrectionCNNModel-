# Financial Trading with Deep CNN


This python notebook implements a CNN to predict whether to buy,hold or sell stock using a time series to image conversion approach used in the paper listed below. The code is extened off of the ideas in the paper aswell as some deviations from other resources. 

# Data: 

Our code downloads price data with a HTTPS GET request and pulls the stock based on the ticker. 
A CVS file of QQQ stock of one year is providede to visualize all the daily stock prices within that year.

https://finance.yahoo.com/quote/QQQ/history?p=QQQ

# Dependencies: 
 
  * pandas -> library used to extract stock data, 
  * numpy ->  library used to manipulate data 
  * matplotlib -> library used to plot data and images 
  * Sklearn -> library used to evalute model performance, feature selection and normalization of data 
  * tensorflow.keras ->  library used to build CNN model 
  * pandas_ta -> library is used to access their technical indicators 

# Data Generation and Preprocessing: 

QQQ is an ETF that contains 100 of NASDAQ's largest companies stock. A five year QQQ stock data from Yahoo! finance is used for our model and contains daily stock data including; timestamps, open price, high price, low price, close price and volume. This data is then used with 15 technical indicators to extract important stock information like, trend, voltality, momentum and volume to generate a 15x15 image to input into our CNN model. 

To generate our three class labels we use a window size of 11 days to determine the max, min and middle prices during these 11 days. Based on these values we assign the lables of sell(max), buy(min) or hold(middle) respectivly. 

A period of 1-26 days was used to iterated through each indicator and storing data for each period. A feature selection method is then used to select the best indicators with respect to the time period. This is useful to determine which indicators at what time frame are best for our model. 

The x_trainand x_train are then reshaped to a 15x15 image that will be the input to the CNN model. 


# Technical Indicators: 

Technical indicators were use extract import stock information over varations of time frames. These indicators will provide information like, momentum, voltality, trend, overlap and volume of the stock over a certain time frame. In our code we have included; 
  
   Trend indicators: DEMA, CCI,MACD, ROC and EMA 
   
   Volatility indicators: ATR, BB, and KC 
   
   Momentum indicators: WR, RSI, SR, MFI, TRIX,CMO, KST and PPO
   
   Volume indicators: EMV, FI,OBV,CMF
   
   Overlap indicators: HMW, TEMA

the code implements the pandas-ta library to calculate indicators using our yahoo finance dataframe. A feature selection is used to determine the best indicators for our model to create our 15X15 image that will be our input for our cnn. (explain) 


# Framework of Model 

The model architect consists of a a CNN network with an input (804,15,15,3) training dimesnison  and output of (252,3). The CNN model consists of two Conv2d layers, one MaxPool2d layer, three Dropout layers, and two non-linear activation layers Relu and Softmax.  

# Evaluation of Model 

Two types of evaluation were considered for this model. The first was using a precision metric to determine how well the model did predicting each label of "buy", "hold and "sell". Two look at the precision of each class label we uesd  confusion Matrix, TP class counts ( for predicted and true) to evalute how well the model preformed. Precision was used instead of an accuracy metric because of the imbalance between classes. The "sell" label class accounted for almost 80% of class label distribution. 

The second evaluation performend on the model was a financial evaluation to determine how well our model did at making money with the predicted labels. Using the predicted labels to determine whether to "buy", "hold",or "sell" stock. 



# Resources for DCNN:

[Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)

[Stock Buy/Sell Prediction Using Convolutional Neural Network](https://towardsdatascience.com/stock-market-action-prediction-with-convnet-8689238feae3) - Toward data science article

[pandas-ta library](https://github.com/twopirllc/pandas-ta) 


