from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from torchvision import transforms
from django.conf import settings

imageClassList = {5: 'bed', 35: 'girl', 65: 'rabbit'}  # Сюда указать классы


def scoreImagePage(request):
    return render(request, 'scorepage.html')


def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/' + fileObj.name, fileObj)
    filePathName = settings.MEDIA_URL + filePathName
    modelName = request.POST.get('modelName')
    scorePrediction, img_uri = predictImageData(modelName, '.' + filePathName)
    context = {'scorePrediction': scorePrediction, 'filePathName': filePathName, 'img_uri': img_uri}
    return render(request, 'scorepage.html', context)


def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    resized_img = img.resize((32, 32), Image.ANTIALIAS)
    img_uri = to_data_uri(resized_img)
    input_image = Image.open(filePath)
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)


    sess = onnxruntime.InferenceSession(
    r'C:\Users\annab\Downloads\cifar100_CNN_RESNET20.onnx')  # <-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': to_numpy(input_batch)}))
    score = imageClassList[outputOFModel]

    return score, img_uri


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RG')
    return img


def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG")  # pick your format


    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')