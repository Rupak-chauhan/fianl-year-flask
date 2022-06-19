import os
import numpy as np
from flask import Flask, request, jsonify, render_template,flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pickle
from predictions import prediction

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

UPLOAD_FOLDER = 'files'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app);
cors = CORS(app, resources={"/predict_api": {"origins": "*", 'Access-Control-Allow-Origin':"*"}})

@app.route('/predict_api',methods=['POST'])
def predict_api():

 # code to make a folder and upload files into it
    print("inside prdict api")

    target=os.path.join(UPLOAD_FOLDER)
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")

    file = request.files['myFile']
    file_name = secure_filename(file.filename)
    destination="/".join([target, file_name])
    file.save(destination)
    session['uploadFilePath']=destination

    print("inside predict api destination")
    output = prediction(destination)
    return jsonify(output)

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True)