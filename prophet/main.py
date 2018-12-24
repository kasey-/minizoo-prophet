#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify
import mimetypes
import hashlib
import bz2
import pickle
import lttb

import json
import numpy as np
import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)
CORS(app)
m = Prophet()

@app.route('/prophet/dataset', methods=["POST"])
def dataset():
    fileid = hashlib.sha224(str(request.data).encode('utf-8')).hexdigest()
    dataFromJson = request.get_json()
    dataFromJson.pop()
    dataNPArray = np.array(dataFromJson)
    df = pd.DataFrame({'ds':dataNPArray[:,0],'y':dataNPArray[:,1]})
    m = Prophet()
    m.fit(df)
    dataArchive = bz2.BZ2File("./archive/"+fileid+".data.bz2", 'w')
    modelArchive = bz2.BZ2File("./archive/"+fileid+".model.bz2", 'w')
    pickle.dump(df, dataArchive)
    pickle.dump(m, modelArchive)
    return jsonify({"id":fileid,"import":True,"fit":True})

@app.route('/prophet/<fileid>/predict/<int:periods>', methods=["GET"])
def predict(fileid,periods):
    m = pickle.load(bz2.BZ2File("./archive/"+fileid+".model.bz2", "r"))
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    predict = json.loads(
        forecast[['ds','yhat','yhat_lower','yhat_upper']].to_json()
    ) # json coding decoding ugly hack ...
    return jsonify({"id":fileid,"forecast":predict})

if __name__ == '__main__':
    app.run(debug=True)
