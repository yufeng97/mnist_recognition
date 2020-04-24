import onnxruntime
from flask import Flask
from flask_restful import Resource, Api, reqparse, abort
from flask_restful.utils import cors

from load_model import *
from utils import allowed_file, ONNX_MODEL

app = Flask(__name__)

api = Api(app)
model = None
sess = onnxruntime.InferenceSession(ONNX_MODEL)


class MNISTResource(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("imgB64", type=str, help="image is not base64")
        self.parser.add_argument("image", type=FileStorage,
                                 location='files', help="image cannot be empty")

    @cors.crossdomain(origin='*')
    def post(self):
        args = self.parser.parse_args()
        image = args.get('image')
        imgB64 = args.get('imgB64')
        array = None
        if not image and not imgB64:
            return {"message": "No image or base64 string"}, 200
        elif imgB64:
            try:
                array = convert_b64(imgB64)
            except Exception as e:
                print(e)
                abort(400, message="decode base64 fail.")
        elif image:
            if not image.filename:
                return {"message": "No image selected."}
            if not allowed_file(image.filename):
                return {"message": "The file must be image type."}
            try:
                array = read_image(image)
            except Exception as e:
                print(e)
                abort(400, message="Something wrong")
        result = onnx_predict(sess, array)
        return {"message": 'success',
                "result": str(result)}, 200

    def options(self):
        return {'Allow': '*'}, 200, {'Access-Control-Allow-Origin': '*',
                                     'Access-Control-Allow-Methods': 'HEAD, OPTIONS, GET, POST, DELETE, PUT',
                                     'Access-Control-Allow-Headers': 'Content-Type, Content-Length, Authorization, '
                                                                     'Accept, X-Requested-With , yourHeaderFeild',
                                     }


api.add_resource(MNISTResource, '/MNIST')

if __name__ == '__main__':
    print("Loading ONNX runtime and Flask starting server ...")
    print("Please wait until server has fully started")
    # model = load_model()
    app.run(port=9000)
