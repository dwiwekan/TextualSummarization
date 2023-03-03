from flask import Flask, jsonify, request
from TextualSummarization import TextualSummarization

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/help")
def help():
    return "<p>Help</p>"

@app.route("/predict", methods=["POST"])
def setName():
    if request.method=='POST':
        posted_data = request.get_json()
        text = posted_data['text']
        persentase = posted_data['persentase']
        awal_akhir = posted_data['awal_akhir']
        posisi = posted_data['posisi']
        ts = TextualSummarization()
        summary = ts.predict(text, persentase, awal_akhir, posisi)
        return jsonify(summary)
