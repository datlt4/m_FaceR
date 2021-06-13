from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import os
import cv2
import sys
from utils.retinaface.retinaface import RetinaFace
from utils.retinaface.prior_box import PriorBox
from utils.retinaface.box_utils import decode, decode_landm
from utils.retinaface.py_cpu_nms import py_cpu_nms

from skimage import transform as trans
import json
import base64
import requests


if not os.path.exists("config.json"):
    sys.exit("There is no configuration file.")

CFG = json.load(open("config.json"))
cfg_mnet = CFG["Parameters"]["Face Detection"]["Retina Face"]["cfg_mnet"]

URL = "http://{}:{}{}".format(CFG["Server"]["Face Detection"]
                              ["Host"], CFG["Server"]["Face Detection"]["Port"], CFG["Server"]["Face Detection"]["route"])

# reference facial points, a list of coordinates (x,y)
src1 = np.array(CFG["Parameters"]["Face Detection"]["Retina Face"]["SRC1"], dtype=np.float32)

#<--left
src2 = np.array(CFG["Parameters"]["Face Detection"]["Retina Face"]["SRC2"], dtype=np.float32)

#---frontal
src3 = np.array(CFG["Parameters"]["Face Detection"]["Retina Face"]["SRC3"], dtype=np.float32)

#-->right
src4 = np.array(CFG["Parameters"]["Face Detection"]["Retina Face"]["SRC4"], dtype=np.float32)

#-->right profile
src5 = np.array(CFG["Parameters"]["Face Detection"]["Retina Face"]["SRC5"], dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(CFG["Parameters"]["Face Detection"]["Retina Face"]["Arcface_src"], dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

"""
    Align image that contained face region based on face landmark was detected.
    Args:
        img (ndarray): image in BGR format.
        landmark (ndarray): landmarks of face detected in size (5, 2)
    Return:
        (ndarray): aligned in BGR format.
"""
def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, _ = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class MRetinaFace:
    def __init__(self, model_dir):
        cudnn.benchmark = True
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        self.model = self.load_model(model_dir).to(self.device)
        self.model.eval()

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(
            used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model_dir):
        # print('Loading pretrained model from {}'.format(model_dir))
        model = RetinaFace(cfg=cfg_mnet, phase='test')

        if torch.cuda.is_available():
            pretrained_dict = torch.load(
                model_dir, map_location=lambda storage, loc: storage.cuda(self.device))
        else:
            pretrained_dict = torch.load(
                model_dir, map_location=lambda storage, location: storage)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        # print('Finished loading model!')
        return model

    # img_raw in BRG format in size (h, w, 3)
    def inference(self, img_raw, confidence_threshold=0.9):
        # default parameters
        top_k = 5000
        keep_top_k = 750
        nms_threshold = 0.4
        resize = 1

        raw_h, raw_w, _ = img_raw.shape
        resized_h = CFG["Parameters"]["Face Detection"]["Retina Face"]["Height Frame"]
        _ratio = float(raw_h) / resized_h
        resized_w = int(float(raw_w) / _ratio)
        resizedImg = cv2.resize(img_raw, dsize=(resized_w, resized_h))

        img = np.float32(resizedImg)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(
            0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )

        # bounding_boxes:   [[x1, y1, x2, y2]]
        # landmarks     :   [[x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]]
        # confidences   :   [c]
        bounding_boxes = np.ones_like(dets[:, :4])
        for i, d in enumerate(dets[:, :4]):
            _x1, _y1, _x2, _y2 = d
            x1 = (1. + CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"]) * \
                _x1 - CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"] * _x2
            x1 = 0 if x1 < 0 else x1
            x2 = (1. + CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"]) * \
                _x2 - CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"] * _x1
            x2 = im_width if x2 > im_width else x2
            y1 = (1. + CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"]) * \
                _y1 - CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"] * _y2
            y1 = 0 if y1 < 0 else y1
            y2 = (1. + CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"]) * \
                _y2 - CFG["Parameters"]["Face Detection"]["Retina Face"]["Padding Face"] * _y1
            y2 = im_height if y2 > im_height else y2
            bounding_boxes[i] = np.array([x1, y1, x2, y2])
        # bounding_boxes = bounding_boxes.astype(np.uint32)
        tl = bounding_boxes[:, :2]
        tl = np.tile(tl.reshape((-1, 2, 1)), 5)
        # landmarks = landms.astype(np.uint32).reshape((-1, 2, 5)) - tl
        landmarks = landms.reshape((-1, 2, 5)) - tl
        confidences = dets[:, -1]
        if len(bounding_boxes) == 0:
            return [], [], []
        else:
            return bounding_boxes * np.array([_ratio, _ratio]*2),\
                landmarks * np.array([[_ratio] * 5, [_ratio] * 5]),\
                confidences


if __name__ == "__main__":
    from flask import Flask, request, Response
    def flask_micro_service():
        app = Flask(__name__)
        mRF = MRetinaFace(CFG["Model Zoo"]["Face Detection"]["Retina Face"]["Pytorch"])

        image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        for _ in range(100): mRF.inference(image)

        @app.route(CFG["Server"]["Face Detection"]["route"], methods=['POST'])
        def m():
            r = request
            if r.method == "POST":
                image_data = base64.decodebytes(r.data)
                nparr = np.fromstring(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    # bounding_boxes:   [[x1, y1, x2, y2]]
                    # landmarks     :   [[x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]]
                    # confidences   :   [c]
                    bounding_boxes, landmarks, confidences = mRF.inference(
                        img_raw=img)
                    response = {"bounding_boxes": bounding_boxes.tolist(),
                                "landmarks": landmarks.tolist(), "confidences": confidences.tolist()}
                    response_pickled = json.dumps(response)
                else:
                    response = {"bounding_boxes": None,
                                "landmarks": None, "confidences": None}
                    response_pickled = json.dumps(response)
            else:
                response = {"bounding_boxes": None,
                            "landmarks": None, "confidences": None}
                response_pickled = json.dumps(response)

            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Face Detection"]["mimetype"])

        app.run(host="0.0.0.0", port=CFG["Server"]["Face Detection"]["Port"])


if __name__ == "m_retina_face":
    def request_micro_service(image):
        _, np_img = cv2.imencode(
            CFG["Server"]["Face Detection"]["encode type"], image)
        byte_img = np_img.tostring()
        base64_bytes = base64.b64encode(byte_img)
        response = requests.post(url=URL, data=base64_bytes,
                                headers=CFG["Server"]["Face Detection"]["Headers"])
        if response.ok:
            r = response.json()
            return np.array(r["bounding_boxes"]), np.array(r["landmarks"]), np.array(r["confidences"])
        else:
            return None


if __name__ == "__main__":
    flask_micro_service()

    # import cv2
    # from m_retina_face import request_micro_service, norm_crop
    # img = cv2.imread("Faces.jpg")
    # bboxs, lmdks, confs = request_micro_service(img)
    # for bbox, lmdk, conf in zip(bboxs, lmdks, confs):
    #     x1, y1, x2, y2 = bbox
    #     aligned = norm_crop(img[int(y1):int(y2), int(x1):int(x2)], lmdk.T)
    #     cv2.imshow("M", aligned)
    #     cv2.waitKey()

