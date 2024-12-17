from flask import Flask, render_template, request, redirect
import datetime
import subprocess
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
UPLOAD_FOLDER = './inputimages/'

@app.route("/", methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		if 'file' not in request.files:
			print('Файл не обнаружен')
			return redirect(request.url)

		file = request.files['file']
		if file.filename == '':
			print('Файл не обнаружен')
			return redirect(request.url)

		file_extension = ".jpg"
		file_name = str(createFileName()) + file_extension
		fpath = UPLOAD_FOLDER + file_name
		file.save(fpath)

		recognize_process = subprocess.run(["./venv/bin/python3", "Recognition.py", file_name], capture_output=True, text=True)
		result = str(recognize_process.stdout)

		return render_template("index.html", img=fpath, result=result)
	else:
		return render_template("index.html")

@app.route("/about")
def about():
	return render_template("niger.html")


def createFileName():
	return datetime.datetime.now()

if __name__ == '__main__':
	app.run(debug=True)
