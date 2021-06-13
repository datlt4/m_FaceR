from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import json
from numpy.core.numeric import indices
import requests

if not os.path.exists("config.json"):
    sys.exit("There is no configuration file.")

CFG = json.load(open("config.json"))

if __name__ == "m_faiss":
    def add(vectors, ids=[]):
        if len(ids) == 0:
            ids = [""] * len(vectors)
        data = {"vectors": vectors.tolist(), "id": ids}
        json_data = json.dumps(data)
        response = requests.post(url="http://{}:{}{}".format(CFG["Server"]["Faiss"]["Host"],
                                 CFG["Server"]["Faiss"]["Port"], CFG["Server"]["Faiss"]["route"]["add"]),
                                 data=json_data, headers=CFG["Server"]["Faiss"]["Headers"])
        if response.ok:
            r = response.json()
            return True
        else:
            return False

    def search(q, k):
        data = {"q": q.tolist(), "k": k}
        json_data = json.dumps(data)
        response = requests.post(url="http://{}:{}{}".format(CFG["Server"]["Faiss"]["Host"],
                                 CFG["Server"]["Faiss"]["Port"], CFG["Server"]["Faiss"]["route"]["search"]),
                                 data=json_data, headers=CFG["Server"]["Faiss"]["Headers"])
        if response.ok:
            r = response.json()
            return np.array(r["distances"]), np.array(r["indices"]), r["id"]
        else:
            return False

    def remove_ids(ids):
        data = {"ids": indices if isinstance(
            indices, list) else indices.tolist()}
        json_data = json.dumps(data)
        response = requests.post(url="http://{}:{}{}".format(CFG["Server"]["Faiss"]["Host"],
                                 CFG["Server"]["Faiss"]["Port"], CFG["Server"]["Faiss"]["route"]["remove_ids"]),
                                 data=json_data, headers=CFG["Server"]["Faiss"]["Headers"])
        if response.ok:
            r = response.json()
            return True
        else:
            return False

    def remove_indices(indices):
        data = {"indices": indices if isinstance(
            indices, list) else indices.tolist()}
        json_data = json.dumps(data)
        response = requests.post(url="http://{}:{}{}".format(CFG["Server"]["Faiss"]["Host"],
                                 CFG["Server"]["Faiss"]["Port"], CFG["Server"]["Faiss"]["route"]["remove_indices"]),
                                 data=json_data, headers=CFG["Server"]["Faiss"]["Headers"])
        if response.ok:
            r = response.json()
            return True
        else:
            return False

    def func(route, data={"Minh": "2210"}):
        response = requests.post(url="http://{}:{}{}".format(CFG["Server"]["Faiss"]["Host"],
                                 CFG["Server"]["Faiss"]["Port"], CFG["Server"]["Faiss"]["route"][route]),
                                 data=json.dumps(data), headers=CFG["Server"]["Faiss"]["Headers"])
        if response.ok:
            r = response.json()
            return r["ntotal"] if route == "ntotal" else r["Minh"]
        else:
            return False

    def ntotal():
        return func("ntotal")

    def reset():
        return func("reset")

    def dumps(filename=CFG["Server"]["Faiss"]["save_dir"]):
        return func("dumps", data={"filename": filename})

    def loads(filename=CFG["Server"]["Faiss"]["save_dir"]):
        return func("loads", data={"filename": filename})

if __name__ == "__main__":
    import faiss
    from flask import Flask, request, Response

    ID_List = []
    gpu = CFG["Parameters"]["Vector Searching"]["Faiss"]["GPU"]

    if gpu is None:
        search_model = faiss.IndexFlatL2(
            CFG["Parameters"]["Vector Searching"]["Faiss"]["Vector Dims"])
    else:
        res = faiss.StandardGpuResources()  # use a single GPU
        # Using a flat index
        index_flat = faiss.IndexFlatL2(
            CFG["Parameters"]["Vector Searching"]["Faiss"]["Vector Dims"])  # build a flat (CPU) index
        # make it a flat GPU index
        search_model = faiss.index_cpu_to_gpu(res, gpu, index_flat)

    app = Flask(__name__)

    @app.route(CFG["Server"]["Faiss"]["route"]["add"], methods=['POST'])
    def add():
        r = request
        if r.method == "POST":
            json_data = json.loads(r.data.decode("utf-8"))
            vectors = np.array(json_data["vectors"], dtype=np.float32)
            search_model.add(vectors)
            if "id" in json_data.keys():
                ID_List.extend(json_data["id"])
            response_pickled = json.dumps({"Minh": True})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    @app.route(CFG["Server"]["Faiss"]["route"]["reset"], methods=['POST'])
    def reset():
        r = request
        if r.method == "POST":
            search_model.reset()
            response_pickled = json.dumps({"Minh": True})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    @app.route(CFG["Server"]["Faiss"]["route"]["remove_ids"], methods=['POST'])
    def remove_ids():
        r = request
        if r.method == "POST":
            ids = json.loads(r.data)["ids"]
            for _id in ids:
                while True:
                    try:
                        index = ID_List.index(_id)
                    except ValueError:
                        break
                    ID_List.pop(index)
                    search_model.remove_ids(np.array([index], dtype='int64'))
            response_pickled = json.dumps({"Minh": True})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    @app.route(CFG["Server"]["Faiss"]["route"]["remove_indices"], methods=['POST'])
    def remove_indices():
        r = request
        if r.method == "POST":
            indices = json.loads(r.data)["indices"]
            indices = sorted(set(indices), reverse=True)
            search_model.remove_ids(np.array([indices], dtype='int64'))
            for idx in indices:
                ID_List.pop(idx)
            response_pickled = json.dumps({"Minh": True})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])


    @app.route(CFG["Server"]["Faiss"]["route"]["search"], methods=['POST'])
    def search():
        r = request
        if r.method == "POST":
            data = json.loads(r.data.decode("utf-8"))
            q = np.array(data["q"], dtype=np.float32)
            k = int(data["k"])
            distances, indices = search_model.search(q, k)
            response_pickled = json.dumps(
                {"distances": distances.tolist(), "indices": indices.tolist(), "id": [[ID_List[j] for j in i] for i in indices.tolist()]})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    @app.route(CFG["Server"]["Faiss"]["route"]["ntotal"], methods=['POST'])
    def ntotal():
        r = request
        if r.method == "POST":
            response_pickled = json.dumps({"ntotal": search_model.ntotal})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"ntotal": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    @app.route(CFG["Server"]["Faiss"]["route"]["loads"], methods=['POST'])
    def loads():
        r = request
        global ID_List
        if r.method == "POST":
            global search_model
            json_data = json.loads(r.data.decode("utf-8"))
            filename = json_data["filename"]
            if gpu is None:
                search_model = faiss.read_index(filename)
            else:
                index_flat = faiss.read_index(filename)
                search_model = faiss.index_cpu_to_gpu(res, gpu, index_flat)

            with open(filename.replace(".bin", ".txt"), mode="r", encoding="utf-8") as f:
                ID_List = f.read().splitlines()

            if len(ID_List) != search_model.ntotal:
                response_pickled = json.dumps({"Minh": False})
            else:
                response_pickled = json.dumps({"Minh": True})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    @app.route(CFG["Server"]["Faiss"]["route"]["dumps"], methods=['POST'])
    def dumps():
        r = request
        if r.method == "POST":
            json_data = json.loads(r.data.decode("utf-8"))
            filename = json_data["filename"]
            filename = filename if ".bin" in filename else filename + ".bin"
            if gpu is None:
                faiss.write_index(search_model, filename)
            else:
                faiss.write_index(
                    faiss.index_gpu_to_cpu(search_model), filename)

            with open(filename.replace(".bin", ".txt"), mode="w", encoding="utf-8") as f:
                f.write("\n".join(ID_List))

            response_pickled = json.dumps({"Minh": True})
            return Response(response=response_pickled, status=200, mimetype=CFG["Server"]["Faiss"]["mimetype"])
        else:
            response_pickled = json.dumps({"Minh": False})
            return Response(response=response_pickled, status=404, mimetype=CFG["Server"]["Faiss"]["mimetype"])

    app.run(host="0.0.0.0", port=CFG["Server"]["Faiss"]["Port"])
