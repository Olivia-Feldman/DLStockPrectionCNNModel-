# Financial Trading with Deep CNN


This python notebook implements a CNN to predict whether to buy,hold or sell stock using a time series to image conversion approach used in the paper listed below. The code is extened off of the ideas in the paper aswell as some deviations from other resources. 

# Data: 

Our code downloads price data with a HTTPS GET request and pulls the stock based on the ticker. 
A CVS file of QQQ stock of one year is providede to visualize all the daily stock prices within that year.

https://finance.yahoo.com/quote/QQQ/history?p=QQQ

# Dependencies: 
 
  * pandas -> used to extract stock data, 
  * numpy ->  used to manipulate data 
  * matplotlib -> used to plot data and images 
  * Sklearn -> used to evalute model performance, feature selection and normalization of data 
  * tensorflow.keras -> used to build CNN model 

# Data Generation and Preprocessing: 

QQQ is an ETF that contains 100 of NASDAQ's largest companies stock. A five year QQQ stock data from Yahoo! finance is used for our model and contains daily stock data including; timestamps, open price, high price, low price, close price and volume. This data is then used with 15 technical indicators to extract important stock information like, trend, voltality, momentum and volume to generate a 15x15 image to input into our CNN model. 

To generate our three class labels we use a window size of 11 days to determine the max, min and middle prices during these 11 days. Based on these values we assign the lables of sell(max), buy(min) or hold(middle) respectivly. 

A period of 1-26 days was used to iterated through each indicator and storing data for each period. A feature selection method is then used to select the best indicators with respect to the time period. This is useful to determine which indicators at what time frame are best for our model. 

The x_test and x_train are then reshaped to a 15x15 image that will be the input to the CNN model. 


# Technical Indicators: 

Technical indicators were use extract import stock information over varations of time frames. These indicators will provide information like, momentum, voltality, trend, overlap and volume of the stock over a certain time frame. In our code we have included; 
  
   Trend indicators: DEMA, CCI,MACD, ROC and EMA 
   
   Volatility indicators: ATR, BB, and KC 
   
   Momentum indicators: WR, RSI, SR, MFI, TRIX,CMO, KST and PPO
   
   Volume indicators: EMV, FI,OBV,CMF
   
   Overlap indicators: HMW, TEMA

the code implements the pandas-ta library to calculate indicators using our yahoo finance dataframe. A feature selection is used to determine the best indicators for our model to create our 15X15 image that will be our input for our cnn. (explain) 


# Resources for DCNN:

[Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)

[Stock Buy/Sell Prediction Using Convolutional Neural Network](https://towardsdatascience.com/stock-market-action-prediction-with-convnet-8689238feae3) - Toward data science article

[pandas-ta library] https://github.com/twopirllc/pandas-ta


