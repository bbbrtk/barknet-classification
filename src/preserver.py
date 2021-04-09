import io, os
import json
import torch, torchvision
from torch.autograd import Variable
from torchvision.transforms import *
import torchvision.transforms as transforms
from PIL import Image
# from flask import Flask, jsonify, request


species = {
'BOJ': 'Betula alleghaniensis',
'BOP': 'Betula papyrifera',
'CHR': 'Quercus rubra',
'EPB': 'Picea glauca',
'EPN': 'Picea mariana',
'EPO': 'Picea abies',
'EPR': 'Picea rubens',
'ERB': 'Acer platanoides',
'ERR': 'Acer rubrum',
'ERS': 'Acer saccharum',
'FRA': 'Fraxinus americana',
'HEG': 'Fagus grandifolia',
'MEL': 'Larix laricina',
'ORA': 'Ulmus americana',
'OSV': 'Ostrya virginiana',
'PEG': 'Populus grandidentata',
'PET': 'Populus tremuloides',
'PIB': 'Pinus strobus',
'PID': 'Pinus rigida',
'PIR': 'Pinus resinosa',
'PRU': 'Tsuga canadensis',
'SAB': 'Abies balsamea',
'THO': 'Thuja occidentalis'
}

classes = [*species]





f= '../data/169_EPO_69_Nexus 5_20170922_102428_7.jpg'
# f = '../data/699_epo_1.jpg'
# app = Flask(__name__)

# params
# classes = ['CHR','EPB','EPO','EPR','ERB','EPN'] # my classes
# classes.sort()
print(classes)

PATH = './log/config/0/resnet34'
CROP_SIZE = 224

def load_model(path):
    net = torch.load(path)
    net.eval()
    return net

def split_crops(img):
    crops = []
    for i in range(img.size[1] // CROP_SIZE):
        for j in range(img.size[0] // CROP_SIZE):
            start_y = i * CROP_SIZE
            start_x = j * CROP_SIZE

            crop = img.crop((start_x, start_y, start_x + CROP_SIZE, start_y + CROP_SIZE))
            crop = ToTensor()(crop)
            crops.append(crop)

    if len(crops) > 0:
        return torch.stack(crops)
    else:
        return []


def get_class_predictions(output):

    predictions = output.max(1)[1]
    predictions = predictions.cpu()

    print('pred:data:', predictions.data)
    # predictions = predictions.data.numpy()
    # predictions = predictions.data.detach().numpy()
    flat_results = predictions.tolist()
    pred = max(set(flat_results), key=flat_results.count)
    return pred

# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([transforms.Resize(255),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             [0.485, 0.456, 0.406],
#                                             [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)


# def get_prediction(model, image_bytes):
#     tensor = transform_image(image_bytes=image_bytes)
#     outputs = model.forward(tensor)
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]

def main():
    model = load_model(PATH)
    img = Image.open(f)
    crops = split_crops(img)
    if len(crops) > 0:
        with torch.no_grad():
            inp = Variable(crops)
            output = model(inp)
            print('output', output)

            pred = get_class_predictions(output)
            code = classes[pred]
            print('pred:\t', pred)
            print('code:\t', code)
            print('name:\t', species[code])

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         img_bytes = file.read()
#         class_id, class_name = get_prediction(image_bytes=img_bytes)
#         return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    main()