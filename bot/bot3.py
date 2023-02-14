import json
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load data from JSON file
with open('C:/Users/H2/Documents/GitHub/QAbot/bot/training_data.json') as file:
    data = json.load(file)

# Extract labeled examples from JSON data
labeled_examples = []
for topic in data['topics']:
    for section in topic['sections']:
        for example in section['examples']:
            labeled_examples.append((example, section['section']))

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define function to create features from tokens
def create_features(tokens):
    features = {}
    for word in tokens:
        features[word] = True
    return features

# Train model
corpus = []
for example in labeled_examples:
    tokens = word_tokenize(example[0].lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    corpus.append((tokens, example[1]))
featuresets = [(create_features(tokens), category) for (tokens, category) in corpus]
model = nltk.NaiveBayesClassifier.train(featuresets)

# Define function to generate chatbot response
def generate_response(user_input):
    tokens = word_tokenize(user_input.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    features = create_features(tokens)
    section = model.classify(features)
    for topic in data['topics']:
        for s in topic['sections']:
            if s['section'] == section:
                response = random.choice(s['responses'])
                return response, model.prob_classify(features).prob(section)

# # Run chatbot loop
# print('Chatbot: Hi, how can I help you today?')
# reward = 0
# while True:
#     user_input = input('User: ')
#     if user_input.lower() == 'exit':
#         print('Chatbot: Goodbye!')
#         break
#     response, confidence = generate_response(user_input)
#     if confidence > 0.5:
#         print('Chatbot:', response, '| Confidence:', confidence)
#         # user_feedback = input('Did I help you? (y/n): ')
#         # if user_feedback.lower() == 'y':
#         #     reward = 1
#         # else:
#         #     reward = -1
#         # Update model using Q-learning or other reinforcement learning algorithm
#         # use a library like TensorFlow or PyTorch to do this
#     else:
#         print("I'm sorry, I'm not sure what you're asking",confidence)


def ask(input : str):
    user_input = input

    response, confidence = generate_response(user_input)
    if confidence > 0.5:
        print('Chatbot:', response, '| Confidence:', confidence)
        # user_feedback = input('Did I help you? (y/n): ')
        # if user_feedback.lower() == 'y':
        #     reward = 1
        # else:
        #     reward = -1
        # Update model using Q-learning or other reinforcement learning algorithm
        # use a library like TensorFlow or PyTorch to do this
    else:
        print("I'm sorry, I'm not sure what you're asking",confidence)
        return response
    
def summarize(input : str):
    user_input = "Summarize" +  input

    response, confidence = generate_response(user_input)
    if confidence > 0.5:
        print('Chatbot:', response, '| Confidence:', confidence)
        return response
        # user_feedback = input('Did I help you? (y/n): ')
        # if user_feedback.lower() == 'y':
        #     reward = 1
        # else:
        #     reward = -1
        # Update model using Q-learning or other reinforcement learning algorithm
        # use a library like TensorFlow or PyTorch to do this
    else:
        print("I'm sorry, I'm not sure what you're asking",confidence)
        return response


summarize("Quantum Mechanics")