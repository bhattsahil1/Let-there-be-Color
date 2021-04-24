import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import subprocess
import sys

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below' +str(filename))
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	# print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/background_process_test/<filename>', methods=['GET', 'POST'])
def background_process_test(filename):
	resize_image = "python3 script.py  -i ./static/uploads/'{0}'".format(filename)
	command_new = "python3 ../colorize.py ./test_image_1.jpg model.pt"
	move_command = "mv '{0}' ./static/".format('result-test_image_1.jpg')
	try:
		resize_ing = subprocess.check_output([resize_image], shell=True)
		result = subprocess.check_output([command_new], shell=True)
		moving = subprocess.check_output([move_command], shell=True)
	except:
		return "Error happened"

	return redirect(url_for('static', filename='result-test_image_1.jpg'), code=301)

if __name__ == "__main__":
	app.run()