import torch
from google_trans_new import google_translator  
import pandas as pd 
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
import torch.nn.functional as F
input_size = 21007
output_size = 2
hidden_size = 17

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, output_size) 
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)        
        return F.log_softmax(x, dim=-1)

model = Net()

def probability(Detect):
    model.load_state_dict(torch.load('/home/vroomer/Documents/API/src/HateSpeech/MODEL.pth'))
    translator = google_translator()  
    translate_text = translator.translate(Detect,lang_tgt='en')
    sent_tokens = sent_tokenize(translate_text)
    sent_tokens = pd.Series(sent_tokens)
    train = pd.read_csv("/home/vroomer/Documents/API/src/HateSpeech/train_tweets.csv")
    X = train['translated']
    Y = train['task_1']
    count_vectorizer = CountVectorizer(min_df=0, max_df=80, ngram_range=(1, 2))
    feature_vector = count_vectorizer.fit_transform(X)
    idx = 0
    sent_tokens = count_vectorizer.transform(sent_tokens).toarray()
    for sent in sent_tokens:
        sample_tensor = torch.from_numpy(sent).float()
        out = model(sample_tensor)
        _, predicted = torch.max(out.data, -1)
        if predicted.item() == 0: 
            return 0
        elif predicted.item() == 1:
            return 1;
        idx+=1