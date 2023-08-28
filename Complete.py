import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import joblib
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api


def process(doc):
    doc = doc.split()
    clean = [re.sub(r'\W+', '', word).lower() for word in doc]
    return clean

def intent_vec(sent):
    size = wv.vector_size
    vec = np.zeros(size)
    ctr = 1
    for word in sent:
        if word in wv:
            ctr+=1
            if(len(wv[word]) != 300):
                print("ERROR with word: {}".format(word)) 
            vec += wv[word]
    vec = vec/ctr
    return vec
    
def intent_ready(list):
    formatted = intent_vec(process(list))
    #formatted = [[np.array(seq) for seq in sent] for sent in formatted]
    return formatted

def intent(sent):
    emp = []
    sent = intent_ready(sent)
    emp.append(sent)
    num = loaded_classifier.predict(emp)
    return num

def response_vec(sent):
    size = wv.vector_size
    wv_res = []
    for word in sent:
        if word in wv:
            if(len(wv[word]) != 300):
                print("ERROR with word: {}".format(word)) 
            wv_res.append(wv[word])
    return wv_res
    
def response_ready(sent):
    formatted = response_vec(process(sent))
    formatted = [np.array(seq) for seq in formatted]
    formatted_with_eos = formatted + [np.zeros(300)]# Add EOS token
    formatted_with_eos = [torch.FloatTensor(seq) for seq in formatted_with_eos]
    return torch.stack(formatted_with_eos)

def decode(encoded):
    decoded_sentence = []
    for vec in encoded:
        end = torch.all(vec <= 0.01)
        if(end):
            break
        
        # Find the closest word vector
        closest_word = wv.similar_by_vector(vec.squeeze(0).numpy(), topn=1)[0][0]
        #print(vec, '\n', closest_word, '\n')
        decoded_sentence.append(closest_word)

    # Assemble the decoded sentence
    decoded_sentence = ' '.join(decoded_sentence)
    return decoded_sentence

def respond(sent):
    num = intent(sent)
    resp = response_ready(sent)
    seq2seq_model.load_state_dict(torch.load(bank[num[0]]))
    with torch.no_grad():
        output = seq2seq_model.generate_response(resp, max_out)
    out = decode(output)
    print("Chatbot Response:", out)

# Define the Encoder-Decoder model
class Seq2Seq(nn.Module):
    def __init__(self, hidden_size):
        super(Seq2Seq, self).__init__()
        
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(300, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(300, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 300)

    def generate_response(self, input_tensor, max_output_length):
        _, hidden = self.encoder(input_tensor.unsqueeze(0))
        decoder_input = torch.zeros(1, 1, 300)  # Start with a zero vector as decoder input
        response = []
        for _ in range(max_output_length):  # Adjust max_output_length based on your needs
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_output = self.fc(decoder_output)
            response.append(decoder_output.squeeze(0))
            decoder_input = decoder_output  # Use decoder_output as the input for the next step

            # Check if the end-of-message token is generated
            end_of_message = torch.all(decoder_output <= 0.1)  # Modify the condition based on your model's behavior

            if end_of_message:
                pad = torch.zeros(1, 300)  # Create a zero tensor with appropriate dimensions
                response.extend([pad] * (max_output_length - len(response)))  # Extend the response list with padding tensors
                break

        response = torch.stack(response)
        return response


bank = ["greetings.pth", "farewells.pth", "info.pth"]
max_out = 50
wv = KeyedVectors.load('google300.kv')
sentence = "goodbye friend"

# Load the model
loaded_classifier = joblib.load("category")


hidden_size = 300
seq2seq_model = Seq2Seq(hidden_size)

while sentence != "stop":
    sentence = input("User Input: ").strip()    
    if (sentence == "stop"):
        break
    respond(sentence)