Personal Project: Two-step Sequence to sequence chatbot


NAME:  < Aaron Verkleeren >


ESTIMATE OF # OF HOURS SPENT ON THIS ASSIGNMENT:  < 50 >

This project was really my first real try at NLP. It is very simple in design and only covers 3 basic intents right now: greetings, information requests, and farewells. 

The first step of this chatbot is a simple logistic regression which takes in text and returns the intent of the message with a 97% accuracy.

The second step is a LSTM sequence to sequence model which can generate responses based on input. I created one for each of the intents and the end result was a seq2seq model that could accurately generate messages based on the input.