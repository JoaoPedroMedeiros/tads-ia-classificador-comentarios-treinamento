from joblib import load

class ClassificadorTextEvaluator:
    """ Class to predict news """
    def __init__(self, text):
        """ Init the class by setting the text

        :param text: The text for classification
        """
        self.text = text
        self.vectorizer = None
        self.model = None
        self.dimension_reducer = None

    def _get_processed_text(self):
        """ Transform and reduce the vectorized text features dimension

        :return: Vectorized text ready to prediction
        """
        transformed = self.vectorizer.transform([self.text])
        return self.dimension_reducer.transform(transformed)

    def _load_model(self):
        """ Loads the previously trained model """
        self.vectorizer = load('resources/vectorizer.joblib')
        self.model = load('resources/token_SVM.joblib')
        self.dimension_reducer = load('resources/dimension_reducer.joblib')

    def predict(self):
        """ Classify the text using the previously trained model

        :return: An array with 1 boolean to say if the new is true or fake
        """
        self._load_model()
        vectorized_news = self._get_processed_text()
        result = self.model.predict(vectorized_news)
        return result
