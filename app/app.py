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


@app.post('/inference')
def inference2(request: Request):

    response_data = {
        'request': request,
        'id': id,
        'predict_results': 'results ...'
    }

    return templates.TemplateResponse('result.html', response_data)



@app.post('/upload')
async def inference(request: Request):
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



