import time
import json
import os

import spacy
# from openai.types import Completion as OpenAICompletion
# from openai import RateLimitError as OpenAIRateLimitError
# from openai import APIError as OpenAIAPIError
# from openai import Timeout as OpenAITimeout
from openai import OpenAI

# from litellm import batch_completion
# from litellm.types.utils import ModelResponse
import boto3

os.environ["OPENAI_API_KEY"] = ""

# Setup spaCy NLP
nlp = None

# Setup OpenAI API
openai_client = None

# Setup Claude 2 API
bedrock = None
anthropic_client = None

os.environ['aws_bedrock_region'] = 'us-west-2'



def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent for sent in doc.sents]


def split_text(text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i + segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def query_gpt4(message):
    client = OpenAI()

    task=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages = task,
        temperature=0,
        max_tokens=4000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    response = response.choices[0].message.content

    return response


def get_claude3_response(prompt, temperature=0,img_path = None, max_new_tokens=300):
    if os.environ.get('aws_bedrock_region'):
        global bedrock
        if not bedrock:
            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.environ.get('aws_bedrock_region')
            )
        img = None
        if img_path:
            img = encode_image(img_path)
        return _get_bedrock_claude_completion(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            base64_image = img,
        )

import base64
# 读取并编码图片
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _get_bedrock_claude_completion(prompt, temperature=0, base64_image = None, max_new_tokens=300):
    while True:
        try:
            if base64_image is None:
                contents = prompt
            else:
                contents = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            messages = [{'role': 'user',  'content': contents }]
            # print(messages)
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            })

            modelId = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
            accept = 'application/json'
            contentType = 'application/json'

            response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

            
            response_body = json.loads(response.get('body').read())

            # print(response_body)

            return response_body['content'][0]['text']
        except Exception as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                time.sleep(10)
                continue
            print(type(e), e)
            return None
        return None
