#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.magface.utils import builder_inf
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import os
import sys
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

class MMagFace():
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = builder_inf(model_path)
        self.model = torch.nn.DataParallel(self.model)
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.]),
        ])
    
    def inference(self, img): # BGR: HxWxC
        with torch.no_grad():
            # compute output
            embedding_feat = self.model(self.trans(img).view(1, 3, 112, 112))
            embedding_feat = F.normalize(embedding_feat, p=2, dim=1)
            _feat = embedding_feat.data.cpu().numpy()[0]
            return _feat


if __name__ == "__main__":
    from flask import Flask, request, Response
    def flask_micro_service():
        app = Flask(__name__)
        mMF = MMagFace(CFG["Model Zoo"]["Face Recognition"]["MagFace"]["Pytorch"])
        image = np.random.randint(0, 256, CFG["Parameters"]["Face Recognition"]["InsightFace"]["Input Shape"], dtype=np.uint8)
        for _ in range(100): mMF.inference(image)
        @app.route(route, methods=['POST'])
        def m():
            r = request
            if r.method == "POST":
                image_data = base64.decodebytes(r.data)
                nparr = np.fromstring(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    embedding = mMF.inference(img)
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


if __name__ == "m_magface":
    def request_micro_service(image):
        _, np_img = cv2.imencode(".png", image)
        byte_img = np_img.tostring()
        base64_bytes = base64.b64encode(byte_img)
        response = requests.post(url=URL, data=base64_bytes, headers={'content-type': 'image/png'})
        if response.ok:
            r = response.json()
            return np.array(r["embedding"])
        else:
            return None


if __name__ == '__main__':
    flask_micro_service()

    

    # import numpy as np
    # mMF = MMagFace("/home/kikai/Documents/AI/MagFace/magface_epoch_00025.pth")
    # img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)

    # from tqdm import tqdm
    # for _ in tqdm(range(10000)):
    #     mMF.inference(img)