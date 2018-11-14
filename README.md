# Motivation
   The stock prediction has been a popular topic for many years but most people approach this problem by analyzing the stock performance in the past to obtain insights about market returns in the future. This approach has been proven to have many shortcomings since company’s performance is not linear and their stock data alone won’t be comprehensive enough to predict their stock movements in the future.
   The objective of our research is taken into account emotional states of a company combine with many quarterly financial KPIs of companies to obtain a more comprehensive stock prediction model.  
   For more explaination of approach to this problem, read `Result Analysis.ipynb`
   
# Code Directory
`Result Analysis.ipynb`
- Explain design of experiments
- Explain Architecture design
- Interpret and compare results of each algorithm
- Conclusion and Future Work

`NLP modeling.ipynb`: 
- Integrate multiple data sources
- Use `nltk` to process text data
- Build Classification model for Sentiment analysis
    
`Generate Sentiment Score for Financial News.ipynb`
- Combining classifier algorithms: each algorithm gets one vote, and the classification that has the votes votes is the chosen one.
- Algorithms voting: tally the votes for and against the winning vote, and call this "confidence."
- Generate sentiment scores as input for Stock Prediction model

`stock_prediction.ipynb`
- Combine sentiment scores and stock data
- Build Classification and Regression model for Stock Prediction

`Crawl_News_Data`
- Write Scrappy crawler to crawl news from Reuter.com
- Clean crawl data in a form that is ready for training
