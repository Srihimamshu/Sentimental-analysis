import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier

def get_words_in_tweets(tweets):
    return [word for (words, _) in tweets for word in words]

def read_tweets(fname, t_type):
    with open(fname, 'r') as f:
        tweets = [[line.strip().split(), t_type] for line in f.readlines()]
    return tweets

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def classify_tweet(tweet):
    return classifier.classify(extract_features(nltk.word_tokenize(tweet)))

# Read in positive and negative training tweets
pos_tweets = read_tweets('happy.txt', 'positive')
neg_tweets = read_tweets('sad.txt', 'negative')

# Preprocess tweets
tweets = []
for sentiment, tweet_list in [(pos_tweets, 'positive'), (neg_tweets, 'negative')]:
    for words, _ in sentiment:
        words_filtered = [word.lower() for word in words if len(word) >= 3]
        tweets.append((words_filtered, tweet_list))

# Extract the word features
word_features = set(get_words_in_tweets(tweets))

# Get the training set and train the Naive Bayes Classifier
training_set = nltk.classify.util.apply_features(extract_features, tweets)
print("Length of training_set:", len(training_set))
print("Sample training instance:", training_set[0])  # Print a sample training instance
print("Number of word features:", len(word_features))

classifier = NaiveBayesClassifier.train(training_set)

# Read in the test tweets and check accuracy
test_tweets = read_tweets('happy_test.txt', 'positive') + read_tweets('sad_test.txt', 'negative')
total = accuracy = len(test_tweets)

for tweet in test_tweets:
    if classify_tweet(tweet[0]) != tweet[1]:
        accuracy -= 1

print('Total accuracy: %f%% (%d/20).' % (accuracy / total * 100, accuracy))
