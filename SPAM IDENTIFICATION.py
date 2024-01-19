# -*- coding: utf-8 -*-


import os
import glob
import pandas as pd
import numpy as np

data = []

# Path for both spam and ham folders
spam_path = '/content/drive/MyDrive/enron1/spam'
ham_path = '/content/drive/MyDrive/enron1/ham'

for file_path in glob.glob(os.path.join(spam_path,'*.txt')):
  with open(file_path,'r',errors = 'replace') as files:
    content = files.read()
    data.append((content,1))

for file_path in glob.glob(os.path.join(ham_path,'*.txt')):
  with open(file_path,'r',errors = 'replace') as files:
    content = files.read()
    data.append((content,0))

data[:5]

# Converting into pandas dataframe
dataframe = pd.DataFrame(data, columns=['emails','labels'])

len(dataframe)

# Removing duplicates
dataframe = dataframe.drop_duplicates(subset='emails')
dataframe.reset_index(drop = True,inplace = True)

len(dataframe)

# Shuffling data to avoid any kind of order
shuffled_data = dataframe.sample(frac = 1, random_state = 42)
shuffled_data

shuffled_data.labels.value_counts()



"""From the above cell we can see that our data is imbalanced. in this case we can consider different
strategies like oversampling or undersampling or any synthetic data generation technique for minority classes. we will handle it later.

"""

# function for visualize some random sample from our dataset
def visualize_text(text):
  import random
  index = random.randint(0,len(shuffled_data)-5)
  for row in text[['emails','labels']][index:index+5].itertuples():
    _,emails,labels = row
    if labels > 0:
      print(f'Label: {labels} (Spam)')
      print(f'text: \n {emails}\n')
      print('-----\n')
    else:
      print(f'Label: {labels} (Not Spam)')
      print(f'Text: \n{emails}\n')
      print('-----\n')
visualize_text(shuffled_data)

# Now visualize some clean data
visualize_text(shuffled_data)

# calculating the average no of words acrros the dataset
sum = 0
for item in shuffled_data['emails']:
  sum = sum + len(item)
avg = sum/len(shuffled_data)
max_length = round(avg)
max_length

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(shuffled_data['emails'],shuffled_data['labels'], test_size = 0.2, random_state = 42)

def pre_process(text):
    import re
    if isinstance(text, str):
        # Only process if the input is a string
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text
    else:
        # Return the input unchanged if it's not a string
        return text

# Removing Stop words
!pip install nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# function for removing stopwords
def remove_stopwords(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

x_train_processed = x_train.apply(pre_process)
x_test_processed = x_test.apply(pre_process)

x_train_processed = x_train_processed.apply(remove_stopwords)
x_test_processed = x_test_processed.apply(remove_stopwords)

vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train_processed)
x_test_vectorized = vectorizer.transform(x_test_processed)

svm_model = make_pipeline(SVC(kernel='linear'))
svm_model.fit(x_train_vectorized, y_train)

# Make predictions on the test set
svm_predictions = svm_model.predict(x_test_vectorized)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))

nb_model = make_pipeline(MultinomialNB())
nb_model.fit(x_train_vectorized, y_train)

# Make predictions on the test set
nb_predictions = nb_model.predict(x_test_vectorized)

# Evaluate the Naive Bayes model
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

from sklearn.linear_model import LinearRegression

!pip install phe

# applying partial homomorphic encryption to our features and labels
from phe import paillier

class Trainer:
    """
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """
    def __init__(self):
        self.model = LinearRegression()

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)



    def fit(self, x_train, y_train):
        self.model = self.model.fit(x_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    # def encrypt_weights(self):
    #     coef = self.model.coef_[0, :]
    #     encrypted_weights = [self.pubkey.encrypt(coef[i])
    #                          for i in range(coef.shape[0])]
    #     encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
    #     return encrypted_weights, encrypted_intercept
    def encrypt_weights(self):
      coef = self.model.coef_
      if len(coef.shape) == 1:  # Handle 1-dimensional case
        encrypted_weights = [self.pubkey.encrypt(coef[i]) for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_)
      else:
        encrypted_weights = [self.pubkey.encrypt(coef[i, :]) for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_)

      return encrypted_weights, encrypted_intercept




    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]


class Tester:


    def __init__(self, pubkey):
        self.pubkey = pubkey

    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def encrypted_score(self, x):
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score

    def encrypted_evaluate(self, X):
        return [self.encrypted_score(text) for text in X]


if __name__ == '__main__':

    print("Trainer: Generating paillier keypair")
    trainer = Trainer()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    trainer.generate_paillier_keypair(n_length=1024)

    trainer.fit(x_train_vectorized,y_train)
    error = np.mean(trainer.predict(x_test_vectorized) != y_test)
    print("Error {:.3f}".format(error))

    print("trainer: Encrypting classifier")

    encrypted_weights, encrypted_intercept = trainer.encrypt_weights()

    print("tester: Scoring with encrypted classifier")
    tester = Tester(trainer.pubkey)
    tester.set_weights(encrypted_weights, encrypted_intercept)

    encrypted_scores = tester.encrypted_evaluate(x_test)

    print("Trainer: Decrypting Testers's scores")

    scores = Trainer.decrypt_scores(encrypted_scores)
    error = np.mean(np.sign(scores) != y_test)
    print("Error {:.3f} -- this is not known to Trainer, who does not possess "
          "the ground truth labels".format(error))

encrypted_weights, encrypted_intercept = trainer.encrypt_weights()


tester.set_weights(encrypted_weights, encrypted_intercept, trainer.pubkey)


encrypted_scores = tester.encrypted_evaluate(X_test_encrypted)


decrypted_scores = trainer.decrypt_scores(encrypted_scores)


from sklearn.metrics import classification_report, confusion_matrix, roc_curve


y_test_np = np.array(y_test)
decrypted_scores_np = np.array(decrypted_scores)


print("Classification Report:")
print(classification_report(y_test_np, decrypted_scores_np))


print("Confusion Matrix:")
print(confusion_matrix(y_test_np, decrypted_scores_np))
class Trainer:
    """
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """
    def __init__(self):
        self.model = SVM(kernel = 'linear', probability = True)

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)



    def fit(self, x_train, y_train):
        self.model = self.model.fit(x_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    # def encrypt_weights(self):
    #     coef = self.model.coef_[0, :]
    #     encrypted_weights = [self.pubkey.encrypt(coef[i])
    #                          for i in range(coef.shape[0])]
    #     encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
    #     return encrypted_weights, encrypted_intercept
    def encrypt_weights(self):
      coef = self.model.coef_
      if len(coef.shape) == 1:  # Handle 1-dimensional case
        encrypted_weights = [self.pubkey.encrypt(coef[i]) for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_)
      else:
        encrypted_weights = [self.pubkey.encrypt(coef[i, :]) for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_)

      return encrypted_weights, encrypted_intercept




    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]


class Tester:


    def __init__(self, pubkey):
        self.pubkey = pubkey

    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def encrypted_score(self, x):
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score

    def encrypted_evaluate(self, X):
        return [self.encrypted_score(text) for text in X]


if __name__ == '__main__':

    print("Trainer: Generating paillier keypair")
    trainer = Trainer()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    trainer.generate_paillier_keypair(n_length=1024)

    trainer.fit(x_train_vectorized,y_train)
    error = np.mean(trainer.predict(x_test_vectorized) != y_test)
    print("Error {:.3f}".format(error))

    print("trainer: Encrypting classifier")

    encrypted_weights, encrypted_intercept = trainer.encrypt_weights()

    print("tester: Scoring with encrypted classifier")
    tester = Tester(trainer.pubkey)
    tester.set_weights(encrypted_weights, encrypted_intercept)

    encrypted_scores = tester.encrypted_evaluate(x_test)

    print("Trainer: Decrypting Testers's scores")

    scores = Trainer.decrypt_scores(encrypted_scores)
    error = np.mean(np.sign(scores) != y_test)
    print("Error {:.3f} -- this is not known to Trainer, who does not possess "
          "the ground truth labels".format(error))

encrypted_weights, encrypted_intercept = trainer.encrypt_weights()


tester.set_weights(encrypted_weights, encrypted_intercept, trainer.pubkey)


encrypted_scores = tester.encrypted_evaluate(X_test_encrypted)


decrypted_scores = trainer.decrypt_scores(encrypted_scores)


from sklearn.metrics import classification_report, confusion_matrix, roc_curve


y_test_np = np.array(y_test)
decrypted_scores_np = np.array(decrypted_scores)


print("Classification Report:")
print(classification_report(y_test_np, decrypted_scores_np))


print("Confusion Matrix:")
print(confusion_matrix(y_test_np, decrypted_scores_np))

fpr, tpr, _ = roc_curve(y_test_np, decrypted_scores_np)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

class Trainer:
    """
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """
    def __init__(self):
        self.model = MultinomialNB()

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)



    def fit(self, x_train, y_train):
        self.model = self.model.fit(x_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    # def encrypt_weights(self):
    #     coef = self.model.coef_[0, :]
    #     encrypted_weights = [self.pubkey.encrypt(coef[i])
    #                          for i in range(coef.shape[0])]
    #     encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
    #     return encrypted_weights, encrypted_intercept
    def encrypt_weights(self):
      coef = self.model.coef_
      if len(coef.shape) == 1:  # Handle 1-dimensional case
        encrypted_weights = [self.pubkey.encrypt(coef[i]) for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_)
      else:
        encrypted_weights = [self.pubkey.encrypt(coef[i, :]) for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_)

      return encrypted_weights, encrypted_intercept




    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]


class Tester:


    def __init__(self, pubkey):
        self.pubkey = pubkey

    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def encrypted_score(self, x):
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score

    def encrypted_evaluate(self, X):
        return [self.encrypted_score(text) for text in X]


if __name__ == '__main__':

    print("Trainer: Generating paillier keypair")
    trainer = Trainer()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    trainer.generate_paillier_keypair(n_length=1024)

    trainer.fit(x_train_vectorized,y_train)
    error = np.mean(trainer.predict(x_test_vectorized) != y_test)
    print("Error {:.3f}".format(error))

    print("trainer: Encrypting classifier")

    encrypted_weights, encrypted_intercept = trainer.encrypt_weights()

    print("tester: Scoring with encrypted classifier")
    tester = Tester(trainer.pubkey)
    tester.set_weights(encrypted_weights, encrypted_intercept)

    encrypted_scores = tester.encrypted_evaluate(x_test)

    print("Trainer: Decrypting Testers's scores")

    scores = Trainer.decrypt_scores(encrypted_scores)
    error = np.mean(np.sign(scores) != y_test)
    print("Error {:.3f} -- this is not known to Trainer, who does not possess "
          "the ground truth labels".format(error))

encrypted_weights, encrypted_intercept = trainer.encrypt_weights()


tester.set_weights(encrypted_weights, encrypted_intercept, trainer.pubkey)


encrypted_scores = tester.encrypted_evaluate(X_test_encrypted)


decrypted_scores = trainer.decrypt_scores(encrypted_scores)


from sklearn.metrics import classification_report, confusion_matrix, roc_curve


y_test_np = np.array(y_test)
decrypted_scores_np = np.array(decrypted_scores)


print("Classification Report:")
print(classification_report(y_test_np, decrypted_scores_np))


print("Confusion Matrix:")
print(confusion_matrix(y_test_np, decrypted_scores_np))

fpr, tpr, _ = roc_curve(y_test_np, decrypted_scores_np)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

