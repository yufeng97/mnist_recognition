import os

from flask import Flask, url_for, render_template, abort, redirect
from flask_uploads import UploadSet, IMAGES, configure_uploads

from utils import UPLOAD_DIR


app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd(), UPLOAD_DIR)  # 文件储存地址
app.config['SECRET_KEY'] = 'demo'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
model = None


@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('upload'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')


@app.route('/upload2')
def upload2():
    return render_template('upload2.html')


@app.route('/hello')
def hello():
    return "Hello World"


if __name__ == '__main__':
    app.run()
