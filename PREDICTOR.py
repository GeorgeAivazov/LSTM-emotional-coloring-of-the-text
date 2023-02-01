import numpy as np
import keras
from keras_preprocessing.sequence import pad_sequences
import re
import pickle



class Downloader:
    def __init__(self):
        pass
    def load_model(self, model_directory, weghts_directory):
        json_file = open(model_directory, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights( weghts_directory)
        return loaded_model
    
    def load_tokenizer(self, tokenizer_directory):
        with open('model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer


class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def preprocessing_text(self,texts):
        texts = re.sub(r'<.*?>', '', texts)
        texts = re.sub(r'[^a-zA-Z]', ' ', texts)
        return ' '.join(x.lower() for x in texts.split())
    
    def predict(self,x):
        x = self.preprocessing_text(x)
        seq = self.tokenizer.texts_to_sequences([x])
        x = pad_sequences(seq, padding='post')
        prediction = self.model.predict(x)
        return prediction.item()
