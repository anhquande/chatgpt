#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
from faker import Faker
from datetime import datetime
import os
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from loaders.ical_loader import ICalLoader

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS
faker = Faker()

pdfLoader = PyPDFLoader("data/mycv.pdf")
textLoader = DirectoryLoader('./data',glob='*.txt')
# icalLoader = ICalLoader('./data/ical')

index = VectorstoreIndexCreator().from_loaders([textLoader, pdfLoader])

@app.route('/v1/completions', methods=['POST'])
def process_prompt():
    # Read the prompt from the request body
    request_json = request.get_json()
    print('requestJson', request_json)
    prompt = request_json['prompt']  # Access prompt using dictionary indexing
    answer = index.query(prompt)

    # Generate a fake response
    response = {
        'id': faker.uuid4(),
        'object': 'text_completion',
        'created': datetime.now().timestamp(),
        'model': 'text-davinci-003',
        'choices': [
            {
                'text': answer,
                'logprobs': None,
                'finish_reason': 'length',
                'index': 0
            }
        ],
        'usage': {
            'prompt_tokens': len(prompt.split()),
            'completion_tokens': 16,
            'total_tokens': len(prompt.split()) + 16
        }
    }

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run()
