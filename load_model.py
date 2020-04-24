from base64 import b64decode
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from onnxruntime import InferenceSession
from torch.nn.functional import log_softmax
from torchvision import transforms
from werkzeug.datastructures import FileStorage

from utils import TARGET_SIZE, TORCH_MODEL, PADDING, BLACK


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(20 * 5 * 5, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        in_size = x.size(0)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        res = conv2.view(in_size, -1)
        out = self.dense(res)
        out = log_softmax(out, dim=1)
        return out


def load_model():
    """
    load model from saved model
    :return: MNIST model
    """
    model = Net()
    model.load_state_dict(torch.load(TORCH_MODEL))
    model.eval()
    return model


def predict(model, tensor: torch.Tensor):
    """
    use model to predict the input value
    :param model: MNIST model
    :param tensor: input tensor
    :return: predict result
    """
    tensor = tensor.view(-1, 1, 28, 28)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.max(output, 1)[1]
    return pred[0].item()


def read_image(image: FileStorage) -> np.ndarray:
    """
    preprocessed the image into designated size tensor
    :param image: origin image (FileStorage type)
    :return: preprocessed ndarray of image
    """
    stream = image.read()
    image = cv2.imdecode(np.asarray(bytearray(stream)), 0)
    resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def convert_base64(string: str) -> torch.Tensor:
    """
    Load base64 string from canvas and convert to tensor
    :param string: base64 string
    :return: preprocessed tensor of image
    """
    x = b64decode(string)
    image = Image.open(BytesIO(x))
    image = image.getchannel(3)
    image = transforms.Resize(TARGET_SIZE)(image)
    tensor = transforms.ToTensor()(image)
    return tensor


def convert_b64(string: str) -> np.ndarray:
    """
    Load base64 string form canvas and convert to ndarray
    :param string: base64 string
    :return: preprocessed ndarray of image
    """
    x = b64decode(string)
    image = cv2.imdecode(np.asarray(bytearray(x)), -1)
    gray = np.copy(image[:, :, -1])
    cropped = crop(gray)
    resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def onnx_predict(sess: InferenceSession, x: np.ndarray):
    """
    use ONNX runtime session to predict result
    :param sess: ONNX runtime session
    :param x: input ndarray
    :return: predicted result
    """
    x = x.reshape((-1, 1, 28, 28))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name: x.astype("float32")})[0]
    return np.argmax(pred, axis=1)[0]


def crop(img: np.ndarray):
    """
    Crop the image to centralise stroke and add some padding
    :param img: image array
    :return: cropped image
    """
    bimg, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (x, y, w, h) = cv2.boundingRect(contours[0])
    cropped = img.copy()[y: y + h, x: x + w]
    if h > w:
        length = h + PADDING
    else:
        length = w + PADDING
    v_pad = (length-h) // 2
    h_pad = (length-w) // 2
    enlarged = cv2.copyMakeBorder(cropped, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=BLACK)
    return enlarged
