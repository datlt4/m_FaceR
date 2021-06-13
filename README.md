# m_FaceR

## 0. Install requirement
```bash
python -m pip install -U mxnet-cu102==1.7 torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 opencv-python opencv-contrib-python Flask faiss-gpu
```


## 1. `dvc`

``` bash
python -m pip install dvc "dvc[s3]" "dvc[gdrive]"
```
or

``` bash
python -m pip install dvc "dvc[all]"
```

### Download models
``` bash
dvc pull
```

## 2. FAISS

### Start `Faiss` microservice

```bash
python m_faiss.py
```

### Request `Faiss` microservice
```python
import m_faiss as F
import numpy as np

m_vectors = np.random.rand(1000, 512)

# add new vectors
status = F.add(m_vectors, ["vec1"]*1000)
status = F.add(np.random.rand(1000, 512), ["vec2"]*1000)
status = F.add(np.random.rand(1000, 512), ["vec3"]*1000)

# get number of vectors
N = F.ntotal()

# delete all vectors
status = F.reset()

# search
query_vector = np.random.rand(10, 512)
distances, indices, ids = F.search(query_vector, k=5)

# remove vectors by ids
F.remove_ids(["vec1", "vec2"])

# remove vectors by index
F.remove_indices([1,2,34,5,6,100,67,7,4,300])

# save database (default save into config.json/Server/Faiss/save_dir)
F.dumps("database/HNIW.bin")

# load saved database (default load from config.json/Server/Faiss/save_dir)
F.loads("database/HNIW.bin")

```

## 3. Retina Face

### Start microservice

```bash
python m_retina_face.py
```

### Request microservice

```python
import cv2
from m_retina_face import request_micro_service, norm_crop

img = cv2.imread("Faces.jpg")
bboxs, lmdks, confs = request_micro_service(img)

for bbox, lmdk, conf in zip(bboxs, lmdks, confs):
    x1, y1, x2, y2 = bbox
    aligned = norm_crop(img[int(y1):int(y2), int(x1):int(x2)], lmdk.T)
```

## 4. InsightFace

### Start microservice

```bash
python m_insightface.py
```

### Request microservice

```python
from m_insightface import request_micro_service
embedding = request_micro_service(aligned_face)
```

## 5. MagFace

### Start microservice

```bash
python m_magface.py
```

### Request microservice

```python
from m_magface import request_micro_service
embedding = request_micro_service(aligned_face)
```

