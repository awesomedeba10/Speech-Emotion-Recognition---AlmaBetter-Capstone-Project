from flask import Flask
from flask.helpers import send_from_directory

app = Flask(__name__, template_folder='html')
app.url_map.strict_slashes = False
app.config.from_object('config')


@app.errorhandler(404)
def not_found(error):
    return {'response': str(error)}, 404

@app.errorhandler(405)
def not_found(error):
    return {'response': str(error)}, 405

from app.modules.recognizer.controller import recognizer_blueprint

app.register_blueprint(recognizer_blueprint)