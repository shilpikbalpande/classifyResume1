import utils
import slate
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings(action = 'ignore')

# my_resume = "C:\\Users\\i847808\\Desktop\\ShilpikBalpande_pdf.pdf"
my_resume = "C:\\Users\\i847808\\Desktop\\\Sample Resume\\HarmishShah.pdf"
# importing all the words from the resumes into a list
resume_list = []
for i in range(4):
    filename = "C:\\Users\\i847808\\Documents\\Resumes\\c" + str(i + 1) + ".pdf"
    # filename = "/c" + str(i + 1) + ".pdf"
    # f = open("CVs" + filename, "rb")
    f = open(filename, "rb")
    doc = slate.PDF(f)
    each_resume = ""
    for j in range(len(doc)):
        each_resume += doc[j]
    resume_list.append(each_resume)

# removing punctuations and other unnecessary characters
for i in range(len(resume_list)):
    resume_list[i] = resume_list[i].translate(string.punctuation)
    resume_list[i] = resume_list[i].translate("\n")
    resume_list[i] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', resume_list[i])

# labelling the existing resumes as being accepted(1) or being rejected(0)
# The first 36 resumes are labelled as accepted in this case and a label list is prepared
label = []
for i in range(1):
    label.append(0)
for i in range(3):
    label.append(1)
label = np.array(label)

# shuffling and splitting the data into a training set and a testing set
resumes_train, resumes_test, y_train, y_test = train_test_split(resume_list, label, test_size=0.33, random_state=42)

# extracting words as features from the training and testing sets and making corresponding feature matrices
vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", max_features=250)
features_train = vectorizer.fit_transform(resumes_train)
X_train = features_train.toarray()
features_test = vectorizer.fit_transform(resumes_test)
X_test = features_test.toarray()

# Using Bernoulli Naive Bayes Algorithm
bnbclf = BernoulliNB()
bnbclf = bnbclf.fit(X_train, y_train)
print(bnbclf.score(X_train, y_train))
print(bnbclf.score(X_test, y_test))



f = open(my_resume, "rb")
sample_resume = slate.PDF(f)
sample_resume = sample_resume[0]
sample_resume = vectorizer.transform([sample_resume])
print(bnbclf.predict(sample_resume)[0])
#print (gnbclf.predict(sample_resume)[0])
