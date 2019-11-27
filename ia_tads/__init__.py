from time import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .evaluation import FakeNewsTextEvaluator

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold

import nltk
from nltk.stem import RSLPStemmer

from .comments_service import CommentsService
from .tokenizer import JuizTokenizer

class JuizTrainer:
    def __init__(self, comments, labels):
        self.start = time()
        self.comments = comments
        self.labels = labels
        self.vectorizer = CountVectorizer(ngram_range=(1, 1), tokenizer=JuizTokenizer())
        self.dimension_reducer = SelectKBest(chi2, k=10350)

    def _get_processed_text(self):
        vectorized_comments = self.vectorizer.fit_transform(self.comments)
        print('Vectorized comments: {:.2f}s'.format(time()-self.start))

        reduced_vectorized_comments = self.dimension_reducer.fit_transform(vectorized_comments, self.labels)

        print('Reduced Vectorized comments: {:.2f}s'.format(time()-self.start))

        return reduced_vectorized_comments

    def _save_model(self, model):
        """ Save the trained model to files

        :param model: model to be saved
        """
        dump(self.vectorizer, 'resources/vectorizer.joblib')
        dump(model, 'resources/token_SVM.joblib')
        dump(self.dimension_reducer, 'resources/dimension_reducer.joblib')

    @staticmethod
    def _train(x_train, y_train):
        """ Trains a SVM and returns it

        :param x_train: features to be used in the training
        :param y_train: labels to be used in the training
        :return: trained SVM
        """
        model = SVC(gamma='auto')
        model.fit(x_train, y_train)
        return model

    def train(self):
        """ Train Aleteia's model and save it to allow future predictions """
        vectorized_comments = self._get_processed_text()

        # print(vectorized_comments)

        model = self._train(vectorized_comments, self.labels)
        predicted_labels = model.predict(vectorized_comments)

        print(accuracy_score(self.labels, predicted_labels))
        print(classification_report(self.labels, predicted_labels, target_names=['negative', 'positive']))
        print(confusion_matrix(self.labels, predicted_labels))

        self._save_model(model)

        print('Model Saved: {:.2f}s'.format(time()-self.start))


nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')

comments = CommentsService.get_labeled_comments()
trainer = JuizTrainer(comments[0], comments[1])
trainer.train()

# evaluator = ClassificadorTextEvaluator('Desgraça')
# result = evaluator.predict()
# print(result)