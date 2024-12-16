from flask import Flask, render_template, request, redirect

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = '/home/leha/plantsrecognizer/webapp/upload/'

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

		fpath = UPLOAD_FOLDER + file.filename
		file.save(fpath)
		print('Файл отправлен')

		return redirect('/')
	else:
		return render_template("index.html")

@app.route("/about")
def about():
	return render_template("niger.html")


if __name__ == '__main__':
	app.run(debug=True)
