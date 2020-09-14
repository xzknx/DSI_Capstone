# Zain Khan: DSI Capstone Project
Title: Can we actually predict market movements by analysing Reddit's /r/wallstreetbets?

## Getting started: 
A full detail analysis and breakdown of my thought process and decisions are on my in-depth blog post here: (https://medium.com/the-innovation/can-we-actually-predict-market-change-by-analyzing-reddits-r-wallstreetbets-9d7716516c8e)
**Data Sources:**
In this project we get our data from 2 sources. 
The first is a publicly available dataset on Kaggle, available here (https://www.kaggle.com/theriley106/wallstreetbetscomments)
The second is acquired by utilizing PushShift (https://pushshift.io/) and Reddit's API PRAW (https://praw.readthedocs.io/).
Please make sure to follow this link (https://www.reddit.com/wiki/api) and the instructions within to get your Reddit API key.

Note: Both data sources are no required for analysis but were rather used for comparison. 

## First and foremost, packages to install:

Datetime:
- datetime

Plotting:
- seaborn
- matplotlib.pyplot

Computational/Data Manipulation:
- numpy
- pandas
- scipy
- coo_matrix
- hstack

Misc:
- tqdm
- json

Sentiment Analysis Tools:
- nltk.sentiment.vader
- codecs
- unidecode
- re
- spacy
- en_core_web_sm
- TfidfVectorizer
- FunctionTransformer
- FeatureUnion

Modelling/Predicting/Classifying:
- StandardScaler
- LinearRegression
- mean_squared_error
- classification_report
- confusion_matrix
- accuracy_score
- plot_confusion_matrix
- train_test_split
- GridSearchCV

- cross_val_score
- LogisticRegression
- LogisticRegressionCV
- RandomForestClassifier
- Pipeline

Yahoo Finance API:
- yfinance

TimeSeries:
- TimeSeriesSplit

## Set up, cleaning, examination and pre-processing:
### The Kaggle dataset:
#### Data preparation:
- The Kaggle dataset was an easy start with data extracted from Reddit /r/wallstreetbets from 2012 up until August 2018. Roughly 2.9m rows of comments extracted from various sources. That was the first issue with the data. We don't exactly know which threads or posts the comments were extracted from. The Kaggle dataset does not include any of that information which meant that this dataset needed to be handled with a pinch of salt. Regardless, it was a great starting point to begin analysis. 
- Cleaning the data and pre-processing was fairly straightforward. Quite simply, remove all of the empty rows and deleted usernames as well. This dataset is fairly simple to work with out of the box. 
- Remove all columns apart from body, score and date. 

#### Sentiment analysis:
- I utilized Vader Sentiment Analysis because it does a great job of working with social media vexicon and, well, that's exactly what these Reddit comments are. 
- Using a for loop, I ran Vader on each row in the body column and appending the respective compound score to a new column in the dataframe. 
- Vader gives us a compound score, negative score and positive score. Initialy I only used the compound score but later, to add more features into our model, I gathered the positive and negative sentiment as well. To add positive and negative sentiment figures to our dataframe, follow the same steps we went through to add the compound score. 
- I ran spaCy on every body text as well so we can run Tf-idf on the text. The cleaned body text was added as a new column to the dataframe.

#### Feature engineering:
- Reddit weighs comments and threads by its score (the sum of upvotes subtracted by the downvotes). This made me think that we should create new columns that weigh the sentiment of each row (positive, negative and compount) according to the score it generates. To do this I simply multiplied each row from the compound_score, positive_score, negative_score (Vader sentiment columns) and score columns. 
- After that I standardized the upvotes (score) column and multiplied each row again to see if the standardization will make a difference to our model. 
- Finally after all that is done, I resample everything on the date index and sum up the numerical values of the Vader scores and upvote scores. 

#### Gathering market data:
- Initially, I tried Quandl but it did not seem to get me up to date information so instead I turned to Yahoo Finance.
- I extracted the SPY prices based on the dates of my dataframe (2012-2018).
- From the SPY prices, I chose the difference on the Close price as the target variable (if Close is higher today than yesterday then it will be positive and vice versa). 
- I shifted the Close column by 1, 2 and 3 as well to use as additional features. 
- The Open, High, Low, Close and Adj Close are used as features too. 

#### Final dataframe:
- To make everything function on one dataframe, I merged everything on the date index.

#### Modelling:
- The Close differenced column is our target (called 'up).
- We split the train and test set on index (Up to 1405 as our training set and onwards for our test set). 
- Create a TimeSeriesSplit with n_splits=7 
- Gather the baseline for our model by calling the value_counts on the target variable and normalizing the scores. 
- GridSearch RandomForrest, DecisionTreeClassifier, KNeighborsClassifier and, of course, LogisticRegression.

#### Results:
- Unsurpringly, LogisticRegression provided the best score while the rest were hovering in and around the baseline for the model anyways. 
- The area under the ROC for LogisticRegression was decent enough but it wasn't convincing because of the reasons below.
- The biggest issues with this dataset and this model was that I wanted to include Robinhood user information from robintrack.net.
- Reddit and /r/wallstreetbets startd to gain more and more momentum once Robinhood was released. It unleashed a massive amount of new investors into the mix. With that in mind, I realized I need to look at extracting data from Reddit starting from the end date of this dataset until August 1st 2020. 
- Couple all the above with the fact that /r/wallstreetbets started to get some real attention in the last 12-18 months, my thoughts were justified. While this dataset provided great insight into analyzing Reddit and setting up the strucutre, the real analysis will begin once we get to gather data from the threads that include serious discussions (like Daily Discussions) rather than, as they say, shitposts. 

#### The LR scores were above the baseline while others fell shockingly short. As expected, I suppose. A few key considerations:
- There is an opportunity here to use PCA to remove multidimensionality between correlated variables (unsupervised linear dimensionality reduction algorithm to find a more meaningful basis or coordinate system for our data). PCA is especially useful if we find that some variables are strongly correlated but can’t necessarily tell which ones to remove.
- I question my decision to run TFIDF on each comment and whether or not that was necessary considered we are only interested with the positive or negative sentiment.
There are more models that I could have tried and implemented to see if the results would be any better or more significant. Support Vector Machines is one I wish - I had more time to try. The same goes for Naive Bayes.
- We did not include any Robinhood user data as a feature in these models. The dates preceded Robinhood and thus it did not make sense to include it at all. There was a short period of overlap between my data and Robinhood user holdings, which I will touch upon later in this article.
- Going forward, I would like to change my target variable from the Dow Close to the SPY Close because I believe a full market ETF would give me a better response to the diversity of conversations on WSB.
- What I have not written in the article is the painstaking amount of time I spent trying to extract individual tickers from each comment from 2012 using Regex and a stock list of the most traded Robinhood stocks by iterating through each comment one by one and assigning each a VADER compound sentiment score. Oh well, as I said before, I am also a self-learning machine. I will only get better.

## Reddit Data Through PushShift and Praw:

#### Scraping and gathering data:
- I use a combination os PushShift and PRAW to extract the data I want for analysis. Rather than take in all the comments from Reddit, I specify which threads and dates I want the data from. 
- Then extract all the individual comments from the threads and addd them to a new column in the dataframe. 
- Just as we did above, run VADER through each and every comment and extract 3 scores for our 3 new columns (compound, positive and negative). 
- With our dataframe in place, I decided to get post titles as well. Reddit has the ability for users to attach a flair to the post title so if the flair is 'Discussion', 'Fundamentals' or 'Stocks', I'll keep it in the dataframe. 
- Once again, run VADER through the titles and get the scores.

#### Market data:
- I gathered market data just as I did above with my new dates. 

#### Feature Engineering:
- I crafted a bullish and bearish vocabulary list to gauge how our post titles align. If we find a word in the title match the bullish list, we append a score of 1 to our new bullish column. The same goes for bearish scores. 
- All of the scores, whether it is VADER, or the bullish and bearish scores are quite volatile and make it hard to understand and visualize movements and changes so this is when I discovered something marvelous from another amazing UVa grad (Arjun Rohling Das), Fourier Transformation (mathematical transformation that decomposes a function into its constituent frequencies).
- I Fourier transformed just about everything I had. I ran Fourier 5, 10, 20 and 30 which are basically different strengths of the transformation (30 giving us more movement and 5 staying relatively more flat).
- After Fourier transformation, I wasn't done, I normalized the Fourier columns, bullish score, bearish score with the MinMaxScaler. I also took the log of the Close price. 

#### EDA: 
- After gathering the market data and the different features we have created, I plot them all in various combinations and permutations to understand if there is indeed a visible trend. Luckily, the bullish and bearish scores align with the market as expected.
- The same goes for the compound score. It follows the ticker well to a certain degree. 

#### Modelling:
- We applied the same exact modelling procedures as we did for the Kaggle dataset. 

#### Results:
- Once again, LogisticRegression was our best model by far. The confusion matrix was fantastic and the area under the curve was almost perfect. 

#### Conclusions:
- This capstone project opened my eyes to the massive world of data and the power that I now have to craft presentations, analysis and findings to an audience.
- Data science is, indeed, a science but also an art. The science needs to be sound, tested and validated but what is the point without storytelling? One must be able to communicate the preprocessing, hypotheses, mathematical modelling and analysis to every stakeholder involved. That’s the beauty of what I’ve learned in the last 3 months at General Assembly.
- With regards to my project above, the goal was to understand if we can predict a positive or negative movement based on the sentiment obtained from Reddit’s /r/wallstreetbets. What we found is that we can, to a great degree of confidence, predict a binary movement in the market. While there is a lot more analysis to be done with significant feature engineering (potentially removing SPY’s High, Low and Open prices as well as the Volume), I believe there is a story developing here. - It isn’t necessarily a finished product (most of data science and storytelling never is) but it illustrates the effects that /r/wallstreetbets can have on the market.
- These traders are active, operate in volumes and, growing day by day.

#### Potential Next Steps
- After a short break, re-evaluate the dataset, features and analysis.
- Implement non-dynamical and dynamic forecasting.
- Create rolling windows for further analysis.
- Consider further EDA and scrape more up to date data
- Understand the correlation between keeping High, Low and Open prices as features for our modelling.
- Learn more about quantitative analysis in finance.
- Explore further financial algorithms and the way they process inputs.
- Understand if there are new and interesting ratios that I can implement to further facilitate my analysis and story.
