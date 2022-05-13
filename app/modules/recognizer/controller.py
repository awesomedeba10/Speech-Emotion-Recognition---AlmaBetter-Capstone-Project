from flask import Blueprint, redirect, request, render_template, send_file, jsonify, json, flash
from glob import glob
from io import BytesIO
from zipfile import ZipFile
import os, glob

from app import app
from app.helper import *

from app.services.load_models import LoadModel

recognizer_blueprint = Blueprint('recognizer', __name__)

@recognizer_blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        uploaded_audio = request.files['audio']
        if allowed_file(uploaded_audio.filename):
            temp_id = return_random(length=14)
            uploaded_audio.filename = temp_id +'.'+ get_file_extension(uploaded_audio.filename)
            audio_save_path = os.path.join(app.config.get('BASE_DIR'), 'temp', uploaded_audio.filename)
            uploaded_audio.save(audio_save_path)

            model = LoadModel(request.form['model'])
            result = model.predict(audio_save_path)

            try:
                os.remove(audio_save_path)
            except:
                pass
            # app.logger.info(result)
            flash("Predicted Emotion for your audio file is : " + result.capitalize(), "success")
            return redirect(url_for('recognizer.index'))
        else:
            flash("File extension not allowed", "error")
            return redirect(url_for('recognizer.index'))


@recognizer_blueprint.route('/download', methods=['GET'])
def download_sample():
    # project_path = os.path.dirname(app.instance_path)
    sample_path = os.path.join(app.config.get('BASE_DIR'), 'data')
    file_list = glob.glob(sample_path + '/**/*.wav', recursive=True)

    stream = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in random.sample(file_list, 10):
            zf.write(file, os.path.basename(file))
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        attachment_filename='sample.zip'
    )