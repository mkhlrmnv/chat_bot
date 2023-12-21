import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# For google api calls
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

import os

import datetime

# nltk.download('punkt')
stemmer = PorterStemmer()

ignore_chars = ['?', '!', ',', '.', '"', "'"]


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def stem_list(sentence):
    result = []
    for w in sentence:
        result.append(stem(w))
    return result


def bag_of_words(sentence, words):
    sentence_words = stem_list(sentence)

    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0

    return bag


def ignore_symbols(list):
    result = []
    for w in list:
        if w not in ignore_chars:
            result.append(w)
    return result


def get_calendar_events():
    # Google scope for calendar
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    # Your google OAuth2.0 .json file
    CLIENT_file = '/Users/mkhlrmnv/Documents/secrets/automation.json'

    creds = None

    # checks if user has already allowed program to use google api
    if os.path.exists('/Users/mkhlrmnv/Documents/secrets/token.json'):
        creds = Credentials.from_authorized_user_file(
            '/Users/mkhlrmnv/Documents/secrets/token.json', SCOPES)

    if not creds or not creds.valid:  # if user haven't done it yet pop-up window will apear where user has to allow program to use google api
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('/Users/mkhlrmnv/Documents/secrets/token.json', 'w') as token:
            token.write(creds.to_json())

    calendar = build('calendar', 'v3', credentials=creds)
    day = datetime.datetime.now().strftime('%y-%m-%d')
    events = calendar.events().list(
        timeMin=f'{datetime.datetime.now().date()}T00:01:00+02:00',
        timeMax=f'{datetime.datetime.now().date()}T23:59:00+02:00',
        calendarId='primary',
    ).execute()

    for event in events['items']:
        print(f"- {event['summary']}")


get_calendar_events()
