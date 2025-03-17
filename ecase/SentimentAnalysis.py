from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch


class SentiementAnalysis:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  
        else:
            self.device = torch.device("cpu")  
        self.model.to(self.device)
        print(self.model.device)


    def sentiment_analysis(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}
        output = self.model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        _p = 0
        _sentiment =  None 
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            if s > _p: 
                _p = s
                _sentiment = l
        return _sentiment