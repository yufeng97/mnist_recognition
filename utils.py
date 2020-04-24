ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_DIR = 'static/uploads'
TORCH_MODEL = "static/MNIST.pkl"
ONNX_MODEL = "static/MNIST.onnx"
TARGET_SIZE = (28, 28)
PADDING = 40
CANVAS_LENGTH = 300
BLACK = (0, 0, 0)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
