import random, string

from flask.helpers import url_for
from flask import request
from urllib.parse import urlencode

def return_random(length=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k = length))

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in __allowed_file()
    )

def get_file_extension(filename):
    return filename.rsplit(".", 1)[1].lower()

def format_error(errMsg):
    return {"status": False, "errors": errMsg}

def __allowed_file():
    return set(['mp3', 'wav'])

def route(name, **query):
    url = url_for(name)
    query_pairs = [(k, vlist) for k,vlist in query.items()]
    return request.url_root + url[1:] + '?'+ urlencode(query_pairs)