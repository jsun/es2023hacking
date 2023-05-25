import os
import io
import string
import shutil
import json
import pathlib
import hashlib
import asyncio
from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import PIL
import numpy as np
import torchvision
import torch
    

APP_ROOT_PATH = pathlib.Path(__file__).resolve().parent

# directory to save the uploaded images
QUEUE_DPATH = os.path.join(APP_ROOT_PATH, 'tasks', 'queue')
# directory to save the processed images
STORAGE_DPATH = os.path.join(APP_ROOT_PATH, 'tasks', 'storage')
# directory to save the model weights
MODEL_WEIGHTS = os.path.join(APP_ROOT_PATH, 'model_weights')


app = FastAPI()
app.mount('/static',
          StaticFiles(directory=os.path.join(APP_ROOT_PATH, 'static')), name='static')
templates = Jinja2Templates(directory=os.path.join(APP_ROOT_PATH, 'templates'))



@app.get('/')
def startup_page(request: Request):
    return templates.TemplateResponse('index.html',
                                      {'request': request, 'id': id})


def image_loader(image_path):
    transform_valid = torchvision.transforms.Compose([
     torchvision.transforms.CenterCrop(224),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
    im = PIL.Image.open(image_path)
    im = transform_valid(im)
    im = im.unsqueeze(0)
    return im
    
    
def get_class_labels(cl_fpath):
    cldict = {}
    with open(cl_fpath, 'r') as f:
        for line in f:
            words = line.rstrip().split('\t')
            if words[0] != '' and words[1] != '':
                cldict[words[1]] = words[0]
    return cldict

    
def crop_model(im_fpath):
    # get class labels
    cldict = get_class_labels('class_labels.txt')
    
    # load models
    crop_app = torchvision.models.resnet18(weights=None)
    num_ftrs = crop_app.fc.in_features
    crop_app.fc = torch.nn.Linear(num_ftrs, len(cldict))
    weight_fpath = '/data/workshop1/pdbidb_1k/meta/resnet18_weights.pth'
    crop_app.load_state_dict(torch.load(weight_fpath))
    crop_app.eval()
    
    # inference
    x = image_loader(im_fpath)
    
    # format outputs
    output = crop_app(x)
    prob = torch.softmax(output, dim=1).detach().numpy().copy().tolist()[0]
    pred_results = []
    for i, p in enumerate(prob):
        pred_results.append({'crop': cldict[str(i)], 'prob': p})
    
    return pred_results
    


# pip install python-multipart
@app.post('/inference')
async def inference(request: Request):
    # get file from HTML form and save it to the directory
    form = await request.form()
    file = form['file']
    model = form['model']
    im_bn = await file.read()
    token = hashlib.md5(im_bn).hexdigest() + '-' + model
    im_fpath = os.path.join(QUEUE_DPATH, token + os.path.splitext(file.filename)[1])
    if not os.path.exists(os.path.join(STORAGE_DPATH, os.path.basename(im_fpath))):
        with open(im_fpath, 'wb') as fh:
            fh.write(im_bn)
    
    pred_result = crop_model(im_fpath)
    
    response_data = {
        'request': request,
        'id': id,
        'uploaded_image': im_fpath,
        'predict_results': pred_result
    }

    return templates.TemplateResponse('result.html', response_data)



@app.post('/upload')
async def upload(request: Request):
    form = await request.form()
    file = form['file']
    model = form['model']
    im_bn = await file.read()
    token = hashlib.md5(im_bn).hexdigest() + '-' + model
    im_fpath = os.path.join(QUEUE_DPATH, model, token + os.path.splitext(file.filename)[1])
    if not os.path.exists(os.path.join(STORAGE_DPATH, os.path.basename(im_fpath))):
        with open(im_fpath, 'wb') as fh:
            fh.write(im_bn)
    return JSONResponse(content = {'token': token})




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



