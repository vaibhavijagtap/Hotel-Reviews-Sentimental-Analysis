# Hotel-Reviews-Sentimental-Analysis

       For luxury hotels, it is very important to overlook the users feedback about their experience in the hotel and its services to improve and avoid negative things and keep the positives. Also, reviews can be analyzed to understand where a hotel failed to provide a good customer experience and where it succeed to do so. Therefore, booking.com has provided this great feature which allows customers to write text reviews about their experience in a hotel, to help the hotels and other costumers.

Overview:
          This project focuses on performing sentiment analysis and prediction on hotel reviews.Our goal is to examine how travelers are communicating their positive and negative experiences on online platforms and major objective is what are the attributes that travelers are considering while selecting a hotel. The goal is to  develop a predictive model to determine the sentiment of future reviews. The sentiment analysis can provide valuable insights for hotel management to improve customer satisfaction and enhance their services.Hre,we build unsupervised Natural Language Processing (NLP) machine learning models that decide whether a text review is positive review or negative review. This project, will help hotels to determine the category of text review and cluster them automaticity to improve their services.

Dataset:

The dataset used in this project is consists of hotel reviews and corresponding ratings.The dataset consists of 20,000 reviews and ratings for different hotels.The dataset contains the following information:
Review: The text of the hotel review.(feedback of travelers)
Rating: The rating given by the customer (ranging from 1 to 5 stars). 
The dataset is preprocessed to handle missing values,handle duplicate value,remove irrelevant information, and perform text preprocessing techniques such as lowercase conversion,emojii removal,digits removal, punctuation removal,Lemmatization and stopwords removal.

Project Overview:

The project follows the following steps: 
1.Data exploration: Understand the structure and characteristics of the hotel review dataset.
2.Data preprocessing: The raw dataset is cleaned and transformed to prepare it for analysis. The cleaning involved removing stopwords, punctuation, and performing tokenization, stemming, or lemmatization. It also includes handling missing values,handling duplicate values, removing irrelevant columns, and performing text preprocessing techniques. 
3.Exploratory Data Analysis (EDA): The preprocessed dataset is analyzed to gain insights into the distribution of ratings, review lengths, and other relevant patterns. Visualizations and statistical summaries are used to explore the data. 
4.Feature Engineering: Extract relevant features from the text data using TF-IDF vectors. Additional features are derived from the text data to capture more information for sentiment analysis. This may include word counts, sentiment scores, or other relevant linguistic features.also we plot the worldcloud for the review.
5. Model Development: Machine learning models, such as Naive Bayes,Logistic Regression,SVC and Random Forest Classifier , are trained on the labeled dataset to classify reviews into positive or negative or neutral sentiments. Various models and algorithms are explored and evaluated for their performance. 
6.Model Evaluation: The trained models are evaluated using appropriate metrics such as accuracy, precision, recall, and F1 score.
7.Deployment:deployment using best model.

Algorithms:
The classification algorithms that has been used in this project on TF-IDF:
Logistic Regression
Multinomial NB
SVC
Random Forest Classifier
Best Algorithm was Logistic Regression TF-IDF with accuracy score 0.83 without overfitting.


Tools:

Python 3.9. Pandas. NumPy. Scikit-learn. Natural Language Toolkit (NLTK). Matplotlib. Jupyter Notebook. Seaborn. Plotly Express.

Results:

The project aims to achieve the following outcomes: Perform exploratory data analysis to gain insights into the hotel reviews dataset. Preprocess the text data by removing stopwords, and applying other text cleaning techniques. Develop and evaluate different machine learning models for sentiment analysis. Select the best-performing model and save it for future predictions. Use the trained model to predict the sentiment of new hotel reviews.

Conclusion:
   Sentiment Analysis is used to classify a review as positive or negative or neutral and best model to get attributes considered in the review by customer is Logistic Regression Model which gave us best accuracy among all models trained.
