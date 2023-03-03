from flask import Flask, jsonify, request
from TextualSummarization import TextualSummarization

data = ''
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/name", methods=["POST"])
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

    
@app.route("/message", methods=["GET"])
def message():
    posted_data = request.get_json()
    data 
    name = posted_data['name']
    return jsonify(" Hope you are having a good time " +  name + "!!!")