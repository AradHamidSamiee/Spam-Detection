import numpy as np
import pandas as pd

# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB

file = pd.read_csv('emails.csv', sep=',') # get elements

# --------------- VARIABLES ---------------
training_data = file.drop(file.columns[0], axis=1)
train = training_data.head(int(len(training_data)*0.8)) # get 80%

testT = training_data.drop(train.index)
test = testT.drop(testT.columns[-1], axis=1)

spamT = train[train['Prediction'] == 1]
spam = spamT.drop(spamT.columns[-1], axis=1)

not_spamT = train[train['Prediction'] == 0]
not_spamA = not_spamT.drop(not_spamT.columns[-1], axis=1)

# GET THE OVERALL SHAPE (ROWS AND COLUMNS)
# print(df.head())
# print(df.shape)
# print(df.columns)

# DROPING DUPLICATES -> NONE DROPED -> NO DUPLICATES
# df.drop_duplicates(inplace = True)
# print(df.shape)

# CHECKING FOR MISSING DATA -> NONE FOUND
# print(df.isnull().sum())

# --------------- VOCAB / PRIOR / LIKELIHOOD ---------------
n_words = len(train.columns) - 1
n_spam = spam.apply(len).sum()

n_not_spamA = not_spamA.apply(len).sum()
p_spam = len(spam) / len(train)
p_not_spamA = len(not_spamA) / len(train)

n_word_spam = spam.sum()
n_word_not_spamA = not_spamA.sum()

p_word_spam = (n_word_spam+1) / (n_spam+n_words)
p_word_not_spamA = (n_word_not_spamA+1) / (n_not_spamA+n_words)

# --------------- MAIN / BAYESIAN LEARNING ---------------
ok = 0
pws = 0 # product of spams
pwo = 0 #product of OKs
# messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])
# X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size=0.20, random=0)
# classifier = MultinomialNB().fit(X_train, y_train)

_SpamWord = list()
OkWord = list()
for k in range(0, len(test)):
    _SpamWord.append(1)
    OkWord.append(1)

for row in range(0, len(test)):
    for column in range(0, len(test.columns)):
        pws += pow(p_word_spam[column], test.iloc[row, column])
        pwo += pow(p_word_not_spamA[column], test.iloc[row, column])
    _SpamWord[row] = np.log(p_spam) + pws
    OkWord[row] = np.log(p_not_spamA) + pwo
    if _SpamWord[row] > OkWord[row] and testT.iloc[row][-1] == 1:
        ok += 1
    if _SpamWord[row] < OkWord[row] and testT.iloc[row][-1] == 0:
        ok += 1

accuracy = (ok / len(test)) * 100

print("OKs:", ok, end="\n")
print("Accuracy:", accuracy, end="\n")
