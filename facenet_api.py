from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from io import BytesIO
import uuid
import numpy as np
import pymysql
import time

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FaceNet models
mtcnn = MTCNN()
model = InceptionResnetV1(pretrained='vggface2').eval()

# MySQL (SingleStore) connection
conn = pymysql.connect(
    host='svc-9cf39a70-a81b-4efd-99ac-1ce82a4c4760-dml.aws-mumbai-1.svc.singlestore.com',
    user='admin',
    password='f5d0wHiB3u0*_2at=SsbVo$',
    database='vector_db',
    autocommit=True
)

@app.post("/upload-face")
async def upload_face(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(BytesIO(contents))
    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected"}

    embedding = model(face.unsqueeze(0)).detach().numpy().flatten()
    embedding_str = ','.join([str(x) for x in embedding])

    face_id = str(uuid.uuid4())
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO face_embeddings (id, embedding) VALUES (%s, VECTOR_ENCODE(?))",
            (face_id, embedding.tolist())
        )
    return {"id": face_id, "message": "Face embedding stored"}

@app.post("/match-face")
async def match_face(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(BytesIO(contents))
    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected"}

    embedding = model(face.unsqueeze(0)).detach().numpy().flatten().tolist()

    start_time = time.time()
    with conn.cursor() as cursor:
        query = """
            SELECT id, DOT_PRODUCT(embedding, VECTOR_ENCODE(%s)) AS similarity
            FROM face_embeddings
            ORDER BY similarity DESC
            LIMIT 1
        """
        cursor.execute(query, (embedding,))
        result = cursor.fetchone()
    end_time = time.time()

    if result:
        return {
            "matched_id": result[0],
            "similarity": float(result[1]),
            "match_time": round(end_time - start_time, 4)
        }
    else:
        return {"message": "No match found"}
    
