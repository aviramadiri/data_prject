
# Final Project - Part B

### Lihi Verchik - 308089333 , Aviram Adiri - 302991468

In this part we will create text classification model depends on the data we collect in part A

### First step: Reading and Organizing the data.

First, we will load the relevnt packages and data.
Then, we will read the data from the file we created in part A:



```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
import pandas as pd
import numpy as np
import matplotlib as plt
from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords
```


```python
df_posts = pd.read_csv("./posts_with_genders.csv",encoding = 'latin1')
```


```python
df_posts.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>Insomniac released a statement about the man's...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>How many people got scammed this weekend?????</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>Idk about yall, but to me the best set was All...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>Did you all know Kaskade performed at EDC yest...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>female</td>
      <td>https://www.facebook.com/jushonti.giberson/pos...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>Bruh! &lt;ed&gt;&lt;U+00A0&gt;&lt;U+00BD&gt;&lt;ed&gt;&lt;U+00B8&gt;&lt;U+00B3&gt;...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>female</td>
      <td>I had a blast being Link on day 2. 70% called ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>I feel bad for everyone still in line for the ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>male</td>
      <td>https://m.facebook.com/story.php?story_fbid=28...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>One of the best time of my life !!! \r\rSee yo...</td>
    </tr>
  </tbody>
</table>
</div>



Then, we got the summary of numerical variables, and plot the histogram of ApplicantIncome.


```python
df_posts.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1092</td>
      <td>1092</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>1067</td>
    </tr>
    <tr>
      <th>top</th>
      <td>male</td>
      <td>Looking for a ride back to the airport that is...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>656</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



We can see that we have lots of unknown chars in our text (emojis and so on).
The next method is used for cleaning the text.


```python
def post_to_words(row_post):
    # Function to convert a raw post to a string of words
    # The input is a single string (a raw post), and 
    # the output is a single string (a preprocessed post review)
    
    # 1. Remove tags
    row_post = re.sub('<.*?>', '', row_post)
    
    # 2. remove non-letters
    letters_only  = re.sub("[^a-zA-Z]", " ", row_post) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()     
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))   
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops] 
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
```

Lets check our method:


```python
clean_post = post_to_words(df_posts["message"][0] )
print(clean_post)
```

    insomniac released statement man death
    

And now let's apply this method on the posts:


```python
clean_posts = df_posts['message'].apply(lambda post: post_to_words(post))
clean_posts.head(10)
```




    0               insomniac released statement man death
    1                      many people got scammed weekend
    2    idk yall best set allison wonderland b b diplo...
    3           know kaskade performed edc yesterday effff
    4    https www facebook com jushonti giberson posts...
    5                                   bruh hope hurt bad
    6                      blast link day called zelda tho
    7                feel bad everyone still line shuttles
    8    https facebook com story php story fbid id us edc
    9      one best time life see next year electronic sky
    Name: message, dtype: object



### bag of words - BOW

The Bag of Words model learns a vocabulary from all of the documents, then models each document by counting the number of times each word appears.


```python
print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 5000) 

# fit_transform() Convert a collection of text documents (reviews in our example) to a matrix of token counts.
# This implementation produces a sparse representation.
# The input to fit_transform should be a list of strings.
posts_data_features = vectorizer.fit_transform(clean_posts)
###train_data_features = vectorizer.fit_transform(train['review'])

# Numpy arrays are easy to work with, so convert the result to an 
# array

posts_data_features = posts_data_features.toarray()
```

    Creating the bag of words...
    
    


```python
vocab = vectorizer.get_feature_names()
print(vocab[0:50])
```

    ['able', 'absolute', 'absolutely', 'ac', 'acceptable', 'access', 'accident', 'accidentally', 'account', 'achey', 'acontext', 'across', 'action', 'actively', 'actually', 'add', 'adding', 'additional', 'ade', 'admiring', 'admission', 'admissions', 'advantage', 'adventure', 'af', 'affordable', 'afrojack', 'afternoon', 'afterwards', 'age', 'agencies', 'ago', 'agrees', 'aid', 'aint', 'air', 'airbrush', 'airport', 'airways', 'albuquerque', 'alcohol', 'alesso', 'alex', 'alexandria', 'alive', 'allison', 'alllove', 'allow', 'allowed', 'allready']
    

### Second step: Training the Model following the train data.
Now, after all the information is completed, we can start with the Training part. 
First, we will split the data to test and train:


```python
import numpy as np

#split to train & test
train_posts = np.random.rand(len(df_posts)) < 0.75
train_message = posts_data_features[train_posts]
train_gender = df_posts.loc[train_posts,"gender"]

test_message = posts_data_features[~train_posts]
test_gender = df_posts.loc[~train_posts,"gender"]
```

Let's check several models:

### First model:  K-Neighbors model.


```python
from sklearn.neighbors import KNeighborsClassifier

KNeighbors = KNeighborsClassifier(n_neighbors=130) 

KNeighbors = KNeighbors.fit( train_message, train_gender )

score = KNeighbors.score(test_message,test_gender )
score
```




    0.61347517730496459



Result - in the best case we got ~ 0.65

### Second model: Gradient Boosting model.


```python
from sklearn.ensemble import GradientBoostingClassifier

GradientBoosting = GradientBoostingClassifier( n_estimators = 45 ) 

GradientBoosting = GradientBoosting.fit( train_message, train_gender )

score = GradientBoosting.score(test_message,test_gender )
score
```




    0.62411347517730498



Result - in the best case we got ~ 0.63

### Third model: Decision Tree model.


```python
from sklearn.tree import DecisionTreeClassifier

DecisionTree= DecisionTreeClassifier(random_state = 1) 

DecisionTree = DecisionTree.fit( train_message, train_gender )

score = DecisionTree.score(test_message,test_gender )
score
```




    0.60992907801418439



Result - in the best case we got ~ 0.62

### 4th model: Logistic Regression model.


```python
from sklearn.linear_model import LogisticRegression

LogisticRegression= LogisticRegression() 

LogisticRegression = LogisticRegression.fit( train_message, train_gender )

score = LogisticRegression.score(test_message,test_gender )
score
```




    0.56737588652482274



Result - in the best case we got ~ 0.56

### 5th model: Random Forest model.


```python
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier= RandomForestClassifier( n_estimators = 80 ) 

RandomForestClassifier = RandomForestClassifier.fit( train_message, train_gender )

score = RandomForestClassifier.score(test_message,test_gender )
score
```




    0.57092198581560283



Result - in the best case we got ~ 0.59

## Summarize
The best result that we got was with K-Neighbors model, with score of 0.65 .

We expected to better results, but we assume that the reason for that is our data- it may be too monotonous and not enugh diverse.

Before the next iteration we will try to replace the data with more diverse data.

