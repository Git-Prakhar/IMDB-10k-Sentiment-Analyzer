import kagglehub
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
nltk.download('stopwords')

# Data Acquisition
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
df = pd.read_csv("./IMDB Dataset.csv")

# Data Preprocessing
df.drop_duplicates(inplace=True)

# Remove HTML tags
def remove_tag(text):
  text = re.sub(re.compile('<.*?>'), '', text)
  return text.lower()
df['review'] = df['review'].apply(remove_tag)

# Remove stopwords
engStopwords = stopwords.words('english')
df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in engStopwords]).apply(lambda x: " ".join(x))

# Post-Processing

# Splitting the data
X = df.iloc[:,0:1]
y = df['sentiment']

# Encoding the target variable
sentimentLabels = LabelEncoder()
y = sentimentLabels.fit_transform(y)

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Using Bag of Words
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

# Training the model using Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)