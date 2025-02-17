import os
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'prepared/intents.json'), encoding='utf-8') as file:
        intents = json.load(file)
        print(intents)
except Exception as e:
    print("Error cargando el JSON:", e)
    raise

words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))

model = load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def sort_key(result):
    """Función para ordenar los resultados por probabilidad."""
    return result[1]


def predict_class(sentence: str) -> list:
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=sort_key, reverse=True)
    return_list = []
    for r in results:
        return_list.append({
            'intent': classes[r[0]], 'probability': str(r[1])
        })
    return return_list


def get_response(intents_list: list, intents_json) -> str:
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = 'No he encontrado algo para decirte'
    for i in list_of_intents:
        if i['tag'] == tag:
            if 'responses' in i and i['responses']:
                result = random.choice(i['responses'])
            else:
                result = 'Lo siento, no tengo una respuesta configurada para esto.'
            break
    return result


print("Vamooooo, el Bot está funsionando")


def call_from_api(message: str) -> str:
    ints = predict_class(message.lower())
    res = get_response(ints, intents)
    return res
