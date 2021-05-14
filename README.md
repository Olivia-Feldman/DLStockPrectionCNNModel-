# HOW TO RUN:

  1. Open demo.ipynb on main branch in Google Colab
      -  This file contains a demo of our 15x15 input CNN-LSTM training model
      -  Also on the main branch are demos of the two other models we tested. They are titled:
             -  demo_LSTM.ipynb
             -  demo_CNN.ipynb
  3. Set model to run on GPU
  4. Run all the cell blocks
  5. Wait patiently for the model to train. 
      -  Graphs of the results can be seen at the bottom of the demo.ipynb file

# Financial Trading with Deep CNN

This python notebook implements a CNN-LSTM to predict whether to buy,hold or sell stock using a time series to image conversion approach similar to the paper listed below. The code is an extension of of the ideas in the paper as well as some deviations from other resources. 

# Data: 

Our code downloads price data with a HTTPS GET request and pulls the stock based on the ticker. 
A CVS file of QQQ stock of one year is provided to visualize all the daily stock prices within that year.

https://finance.yahoo.com/quote/QQQ/history?p=QQQ

# Dependencies: 
 
  * pandas -> library used to extract stock data, 
  * numpy ->  library used to manipulate data 
  * matplotlib -> library used to plot data and images 
  * Sklearn -> library used to evalute model performance, feature selection and normalization of data 
  * tensorflow.keras ->  library used to build CNN model 


# Data Generation and Preprocessing: 

QQQ is an ETF that contains 100 of NASDAQ's largest companies stock. A five year QQQ stock data from Yahoo! finance is used for our model and contains daily stock data including; timestamps, open price, high price, low price, close price and volume. This data is then used with 15 technical indicators to extract important stock information like, trend, voltality, momentum and volume to generate a 15x15 image to input into our CNN model. 

To generate our three class labels we use a window size of 11 days to determine the max, min and middle prices during these 11 days. Based on these values we assign the lables of sell(max), buy(min) or hold(middle) respectivly. 

A period of 1-26 days was used to iterated through each indicator and storing data for each period. A feature selection method is then used to select the best indicators with respect to the time period. This is useful to determine which indicators at what time frame are best for our model. 

The x_train and x_test are then reshaped to a 15x15 image that will be the input to the CNN model. 


# Technical Indicators: 

Technical indicators were use extract import stock information over varations of time frames. These indicators will provide information like, momentum, voltality, trend, overlap and volume of the stock over a certain time frame. In our code we have included; 
  
   Trend indicators: DEMA, CCI,MACD, ROC and EMA 
   
   Volatility indicators: ATR, BB, and KC 
   
   Momentum indicators: WR, RSI, SR, MFI, TRIX,CMO, KST and PPO
   
   Volume indicators: EMV, FI,OBV,CMF
   
   Overlap indicators: HMW, TEMA

 A feature selection is used to determine the best indicators for our model to create our 15X15 image that will be our input for our cnn. (explain) 


# Framework of Model 

The model architect consists of a a CNN network with an input (none,15,15,3) training dimesnison  and output of (252,3). The CNN model consists of two Conv2d layers, one MaxPool2d layer, three Dropout layers, and two non-linear activation layers Relu and Softmax.  

# Evaluation of Model 

Two types of evaluation were considered for this model. The first was using a precision metric to determine how well the model did predicting each label of "buy", "hold and "sell". Two look at the precision of each class label we uesd  confusion Matrix, TP class counts ( for predicted and true). We implemented our own accuracy metric tocorrectly determine the model accuracy for our imbalance class labels.  The "sell" label class accounted for almost 90% of class label distribution. 

The second evaluation performend on the model was a financial evaluation to determine how well our model did at making money with the predicted labels. A trading algorithm was implemented assuming USD currency, that the trading fractional shares were not restricted and their were no broker fees. The maximum trade size was 3000.00 and teh initial cash on hand was $10000.0 and initial number of shares was 0. 


THe model had an accuracy of 74.603% and an financial evaluation  $3000.139 cash left on hand, 27.277 number of shares with a value of $8974.88 and a totla value of $11975.024. 


## Resources used for DL models:

[Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)

[Stock Buy/Sell Prediction Using Convolutional Neural Network](https://towardsdatascience.com/stock-market-action-prediction-with-convnet-8689238feae3) - Toward data science article

[DCNN github with code](https://github.com/nayash/stock_cnn_blog_pub)

[Stock Prediction using Machine Learning and Python](https://www.youtube.com/watch?v=lncoLfue_Y4&ab_channel=edureka%21)

https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/deep-learning/1.lstm.ipynb --> Stock prediction code using LSTM


https://towardsdatascience.com/aifortrading-2edd6fac689d

https://www.ig.com/us/trading-strategies/what-are-the-best-swing-trading-indicators-200421
