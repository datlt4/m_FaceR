from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import numpy as np
import mxnet as mx
import json
import base64
import requests

if not os.path.exists("config.json"):
    sys.exit("There is no configuration file.")

CFG = json.load(open("config.json"))

route = CFG["Server"]["Face Recognition"]["route"]
URL = "http://{}:{}{}".format(CFG["Server"]["Face Recognition"]["Host"],
                              CFG["Server"]["Face Recognition"]["Port"],
                              CFG["Server"]["Face Recognition"]["route"])


class MInsightFace:
    def __init__(self, model_dir):
        self.ctx = self.gpu_device()
        self.get_model(self.ctx, model_dir)

    def gpu_device(self, gpu_number=0):
        try:
            _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
            return mx.gpu()
        except mx.MXNetError:
            return mx.cpu()

    def get_model(self, ctx, model_dir):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_dir, 0)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.model = mx.mod.Module(
            symbol=sym, context=self.ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)
        return self.model

    # face aligned in <BGR> format with shape (112, 112, 3)
    def inference(self, aligned, preprocess=True):
        if preprocess:
            aligned = cv2.resize(aligned, (112, 112))

        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(aligned, (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = embedding.reshape((512,))
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


if __name__ == "__main__":
    from flask import Flask, request, Response
    def flask_micro_service():
        app = Flask(__name__)

        mIF = MInsightFace(CFG["Model Zoo"]["Face Recognition"]["InsightFace"]["MxNet"])
        image = np.random.randint(0, 256, CFG["Parameters"]["Face Recognition"]["InsightFace"]["Input Shape"], dtype=np.uint8)
        for _ in range(100): mIF.inference(image)

        @app.route(route, methods=['POST'])
        def m():
            r = request
            if r.method == "POST":
                image_data = base64.decodebytes(r.data)
                nparr = np.fromstring(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    embedding = mIF.inference(img)
                    response = {"embedding": embedding.tolist()}
                    response_pickled = json.dumps(response)
                else:
                    response = {"embedding": None}
                    response_pickled = json.dumps(response)
            else:
                response = {"embedding": None}
                response_pickled = json.dumps(response)

            return Response(response=response_pickled, status=200, mimetype="application/json")

        app.run(host="0.0.0.0", port=CFG["Server"]["Face Recognition"]["Port"])


if __name__ == "m_insightface":
    def request_micro_service(image):
        _, np_img = cv2.imencode(".png", image)
        byte_img = np_img.tostring()
        base64_bytes = base64.b64encode(byte_img)
        response = requests.post(url=URL, data=base64_bytes, headers={
                                'content-type': 'image/png'})
        if response.ok:
            r = response.json()
            return np.array(r["embedding"])
        else:
            return None


if __name__ == "__main__":
    flask_micro_service()
