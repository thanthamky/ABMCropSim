from flask import Flask, request #เพิ่ม request
from ast import literal_eval
from datetime import datetime

from CropSelectionModel import CropSelector

app = Flask(__name__)
model = CropSelector("DT_model2.pkl", "KNN_model2.pkl", "crop_encoder.pkl")

@app.route('/')
def main():

    data = request.args.get('data')

    error_internal_message = "No error"
    message = 'success'
    status = 0
    dateTimeText = datetime.now()
    success = True

    try:

        data = literal_eval(data)
        result = model.select_crop(data)

    except Exception as e:

        error_internal_message = e
        message = 'Failed execution!'
        status = -1
        success = False


    return { "data": result, "error": {"internalErrorMessage": error_internal_message, "message": message, "status":0, "timestamp": dateTimeText}, "success": True }, 201
  

app.run()