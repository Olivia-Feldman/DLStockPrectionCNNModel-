# HOW TO RUN:

  1. Open ***demo.ipynb*** on main branch in Google Colab
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
  
   Trend indicators: DEMA, CCI,MACD, ROC, EMA 
   
   Volatility indicators: ATR, BB, KC 
   
   Momentum indicators: WR, RSI, SR, MFI
   
   Volume indicators: EMV, FI,OBV
   
  

 A feature selection is used to determine the best indicators for our model to create our 15X15 image that will be our input for our cnn. (explain) 


# Framework of Models

CNN: 

The model architect consists of  training shapes of ()and output of (252,3). The CNN model consists of two Conv2d layers, one MaxPool2d layer, three Dropout layers, and two non-linear activation layers Relu and Softmax. 

LSTM: 
The model architect consists training shapes of (2505,10,7) and (2505,1), and  testing shapes of ( 251,10,7) and (251,1). The training data is then is normalized from zero to one and then modified with a time step of 10. The time step is used to group the time series data before feeding the data into the model. The larger the time step value is, the more data points there are for the LSTM model to look back on to make a prediction from. The  training data is then fed into a LSTM network containing three LSTM layer each followed by a dropout layer and lastly a dense layer.  The ouput is a (2505, 1) array that predicts the close price for the next day from the given training data. 


CNN-LSTM: 

The CNN-LSTM arichitect consists of training shapes consists of (2012,333) and (2012,) and testing shapes of (252,333) and (252,). The training data contains 333 time variations of our techincal indicators from our data preprocessing. This was created by using ANOVA feature selection to determine the best indicators to used for generating the 15x15 images. The training data is also normalized from 0,1 and reshaped into a (2002, 15,15,3) and  (2002,) array and fed into the CNN-LSTM network. The testing data is normalized aswell and rehaped to a final shape of (252,10,15,15,3) and (252,) arrays. The CNN-LSTM network contains two 2D convolution layers, 2D max pooling layer, relu activation layer and is then fed into a LSTM network and dense layer. A time distributed layer is used to apply the conv2d layer to all the each of the time steps individually. The output of the network is (3,) array of probabilites for sell, buy, and hold labels. 


# Evaluation of Models 

CNN: 
Two types of evaluation were considered for this model. The first was using a precision metric to determine how well the model did predicting each label of "buy", "hold and "sell". Two look at the precision of each class label we uesd  confusion Matrix, TP class counts ( for predicted and true). We implemented our own accuracy metric tocorrectly determine the model accuracy for our imbalance class labels.  The "sell" label class accounted for almost 90% of class label distribution. The second evaluation performend on the model was a financial evaluation to determine how well our model did at making money with the predicted labels. A trading algorithm was implemented assuming USD currency, that the trading fractional shares were not restricted and their were no broker fees. The maximum trade size was 3000.00 and teh initial cash on hand was $10000.0 and initial number of shares was 0. The model had an accuracy of 74.603% and an financial evaluation  $3000.139 cash left on hand, 27.277 number of shares with a value of $8974.88 and a totla value of $11975.024. 

LSTM: 

The LSTM model used an accuracy metric that compared the predicted and acutal prices to evalute the model aswell as a trading algorithm. This algorithm was used to evlaute how well the model did at generating a profit by buying and selling stock from the predicted prices. This algorithm consists of finding the change in between the predicted current and previous price. If the percent change is greater than 1% then the model would expect the price to increase, a few shares are bought. If the change is less, then -1.5% then the model would expect the stock price to decrease, and a few shares are sold. The number of shares to be bought or sold is determined the current cash on hand and the actual previous price of the stock. The model is also constrained so that it cannot own less than 10 shares after it begins purchasing and have less than $500 USD dollars at any one time. The maximum trading size is $3,000.00 and the initial cash on hand value is $10,000.00 USD. The initial number of shares is zero. These two metrics were used to evaluate the overall performance of the LSTM model.  


CNN-LSTM: 

This model was evaluted using an accuracy metric, confusion matrix and trading algorithm. The model had a 75.00% accuracy and this was determined by comparing the actual labels to the predicted labels. The confusion matrix was used to examine the TP and NP for the labels. The hold label had great precision of correctly labeling 184 labels, while the buy class labled 1 of 23 labels correctly and the sell label only predicted 3 of 12 of it's lables correctly. A trading algorithm was implemented assuming USD currency, that the trading fractional shares were not restricted and their were no broker fees. The maximum trade size was 3000.00 and teh initial cash on hand was $10000.0 and initial number of shares was 0. The model reported back $8622.736 cash left on hand and 10.421 shares with the total value of the share being $11884.09. 







## Resources used for DL models:

[Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)

[Stock Buy/Sell Prediction Using Convolutional Neural Network](https://towardsdatascience.com/stock-market-action-prediction-with-convnet-8689238feae3) - Toward data science article

[DCNN github with code](https://github.com/nayash/stock_cnn_blog_pub)

[Stock Prediction using Machine Learning and Python](https://www.youtube.com/watch?v=lncoLfue_Y4&ab_channel=edureka%21)

https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/deep-learning/1.lstm.ipynb --> Stock prediction code using LSTM


https://towardsdatascience.com/aifortrading-2edd6fac689d

https://www.ig.com/us/trading-strategies/what-are-the-best-swing-trading-indicators-200421
