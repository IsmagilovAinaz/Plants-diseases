from flask import Flask, render_template, request, redirect
import datetime
import subprocess
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg'}
UPLOAD_FOLDER = './static/img/input/'

@app.route("/", methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		if 'file' not in request.files:
			result='Файл не обнаружен'
			return render_template("index.html", result=result)

		file = request.files['file']
		if file.filename == '':
			result='Файл не обнаружен'
			return render_template("index.html", result=result)

		file_extension = ".jpg"
		file_name = str(createFileName()) + file_extension
		fpath = UPLOAD_FOLDER + file_name
		file.save(fpath)

		recognize_process = subprocess.run(["./venv/bin/python3", "Recognition.py", file_name], capture_output=True, text=True)
		result = str(recognize_process.stdout)

		return render_template("index.html", img=fpath, result=result)
	else:
		return render_template("index.html")

def createFileName():
	return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

if __name__ == '__main__':
	app.run(debug=True)
