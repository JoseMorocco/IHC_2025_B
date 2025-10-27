# -*- coding: utf-8 -*-
"""
Tutorial: Reconocimiento Facial con Machine Learning en Python
Basado en: https://codificandobits.com/tutorial/reconocimiento-facial-machine-learning-python/

Este script implementa un sistema de reconocimiento facial que utiliza:
- MobileNet para la detección de rostros.
- FaceNet para el cálculo de embeddings y verificación facial.

La idea es determinar si en una imagen (o múltiples rostros en una imagen) se encuentra
el rostro de un sujeto en particular, comparándolo con un conjunto de rostros de referencia.
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- 1. Inicialización y funciones de utilidad ---

# Definición de directorios
DIR_KNOWNS = 'knowns'
DIR_UNKNOWNS = 'unknowns'
DIR_RESULTS = 'results'

# Crear directorios si no existen
os.makedirs(DIR_KNOWNS, exist_ok=True)
os.makedirs(DIR_UNKNOWNS, exist_ok=True)
os.makedirs(DIR_RESULTS, exist_ok=True)

# Función para cargar imágenes
def load_image(DIR, name):
    """
    Carga una imagen del directorio especificado y la convierte a formato RGB.
    """
    path = os.path.join(DIR, name)
    img = cv2.imread(path)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Función para dibujar bounding boxes
def draw_box(image, box, color, line_width=6):
    """
    Dibuja un rectángulo sobre la imagen en las coordenadas especificadas.
    """
    if not box: # Si la caja está vacía
        return image
    else:
        # Las coordenadas de box son [left, right, top, bottom]
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), color, line_width)
    return image

# --- 2. Detección de rostros con MobileNet ---

# Cargar el modelo MobileNet pre-entrenado
try:
    with tf.io.gfile.GFile('mobilenet_graph.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as mobilenet_graph:
        tf.import_graph_def(graph_def, name='')
    print("MobileNet cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar MobileNet: {e}")
    print("Asegúrate de que 'mobilenet_graph.pb' está en el mismo directorio.")
    exit() # Salir si el modelo no se carga

# Función para detectar rostros
def detect_faces(image, score_threshold=0.7):
    """
    Detecta rostros en una imagen usando MobileNet y devuelve sus bounding boxes.
    """
    if image is None:
        return []
        
    (imh, imw) = image.shape[:-1]
    img = np.expand_dims(image, axis=0)
    
    # Inicializar sesión de MobileNet
    with tf.compat.v1.Session(graph=mobilenet_graph) as sess:
        image_tensor = mobilenet_graph.get_tensor_by_name('image_tensor:0')
        boxes = mobilenet_graph.get_tensor_by_name('detection_boxes:0')
        scores = mobilenet_graph.get_tensor_by_name('detection_scores:0')
        
        # Predicción (detección)
        (boxes, scores) = sess.run([boxes, scores], feed_dict={image_tensor: img})
    
    # Reajustar tamaños boxes, scores
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    
    # Depurar bounding boxes (filtrar por umbral de confianza)
    idx = np.where(scores >= score_threshold)[0]
    
    # Crear bounding boxes en formato [left, right, top, bottom]
    bboxes = []
    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index, :]
        (left, right, top, bottom) = (xmin * imw, xmax * imw, ymin * imh, ymax * imh)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        bboxes.append([left, right, top, bottom])
        
    return bboxes

# --- 3. Reconocimiento facial con FaceNet ---

# Cargar el modelo FaceNet pre-entrenado
try:
    # MODIFICACIÓN CLAVE AQUÍ: añadir safe_mode=False
    facenet_model = load_model('facenet_keras.h5', safe_mode=False) 
    print("FaceNet cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar FaceNet: {e}")
    print("Asegúrate de que 'facenet_keras.h5' está en el mismo directorio y que el modelo es compatible.")
    exit() # Salir si el modelo no se carga

# Función para extraer rostros de la imagen
def extract_faces(image, bboxes, new_size=(160, 160)):
    """
    Extrae y redimensiona los rostros de una imagen basándose en las bounding boxes.
    """
    cropped_faces = []
    if image is None:
        return []

    for box in bboxes:
        left, right, top, bottom = box
        # Asegurarse de que las coordenadas no excedan las dimensiones de la imagen
        top = max(0, top)
        bottom = min(image.shape[0], bottom)
        left = max(0, left)
        right = min(image.shape[1], right)

        face = image[top:bottom, left:right]
        if face.size > 0: # Asegurarse de que el recorte no esté vacío
            cropped_faces.append(cv2.resize(face, dsize=new_size))
    return cropped_faces

# Función para calcular el embedding de un rostro
def compute_embedding(model, face):
    """
    Calcula el embedding de un rostro usando el modelo FaceNet.
    """
    if face is None or face.size == 0:
        return None

    face = face.astype('float32')
    
    # Estandarización de la imagen
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    
    face = np.expand_dims(face, axis=0) # Añadir dimensión de lote
    
    embedding = model.predict(face, verbose=0) # verbose=0 para suprimir salida
    return embedding[0] # Devolver solo el vector de embedding

# Función para comparar embeddings
def compare_faces(embs_ref, emb_desc, umbral=11):
    """
    Compara un embedding desconocido con un conjunto de embeddings de referencia.
    Retorna las distancias y una lista booleana indicando si hay coincidencia.
    """
    distancias = []
    for emb_ref in embs_ref:
        if emb_desc is not None and emb_ref is not None:
            distancias.append(np.linalg.norm(emb_ref - emb_desc))
        else:
            distancias.append(float('inf')) # Distancia infinita si un embedding es nulo
    
    distancias = np.array(distancias)
    return distancias, list(distancias <= umbral)

# --- 4. Flujo principal del reconocimiento facial ---

# 4.1. Calcular embeddings de referencia (rostros conocidos)
known_embeddings = []
print('Procesando rostros conocidos...')
known_face_names = [] # Para almacenar nombres asociados a embeddings

for name in os.listdir(DIR_KNOWNS):
    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f'   {name}')
        image = load_image(DIR_KNOWNS, name)
        if image is None:
            continue
        bboxes = detect_faces(image)
        if bboxes: # Solo si se detectan rostros
            faces = extract_faces(image, bboxes)
            if faces: # Si se pudieron extraer rostros
                embedding = compute_embedding(facenet_model, faces[0])
                if embedding is not None:
                    known_embeddings.append(embedding)
                    known_face_names.append(os.path.splitext(name)[0]) # Almacenar nombre sin extensión
                else:
                    print(f"       No se pudo calcular el embedding para {name}")
            else:
                print(f"       No se pudieron extraer rostros de {name}")
        else:
            print(f"       No se detectaron rostros en {name}")

if not known_embeddings:
    print("Error: No se pudieron procesar rostros conocidos. Asegúrate de tener imágenes en la carpeta 'knowns'.")
    exit()

# 4.2. Procesar imágenes desconocidas y realizar verificación
print('\nProcesando imágenes desconocidas...')
for name in os.listdir(DIR_UNKNOWNS):
    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f'   {name}')
        image = load_image(DIR_UNKNOWNS, name)
        if image is None:
            continue
        bboxes = detect_faces(image)
        faces = extract_faces(image, bboxes)
        
        img_with_boxes = image.copy()
        
        if not faces:
            print(f"       No se detectaron rostros en {name}. Saltando...")
            # Guardar la imagen original si no se detectaron rostros
            cv2.imwrite(os.path.join(DIR_RESULTS, name), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
            continue

        for face, box in zip(faces, bboxes):
            emb = compute_embedding(facenet_model, face)
            
            if emb is not None:
                _, reconocimiento = compare_faces(known_embeddings, emb)
                
                if any(reconocimiento):
                    print('     match!')
                    img_with_boxes = draw_box(img_with_boxes, box, (0, 255, 0)) # Verde para coincidencia
                else:
                    img_with_boxes = draw_box(img_with_boxes, box, (255, 0, 0)) # Rojo para no coincidencia
            else:
                print(f"       No se pudo calcular embedding para un rostro en {name}.")
                img_with_boxes = draw_box(img_with_boxes, box, (0, 0, 255)) # Azul si hay error en embedding
            
        cv2.imwrite(os.path.join(DIR_RESULTS, name), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

print('\n¡Fin del reconocimiento facial!')

# Opcional: Mostrar un resultado si se ejecuta en un entorno interactivo como Jupyter
# if 'IPython' in globals():
#     # Aquí podrías cargar y mostrar una de las imágenes de resultado si estás en Jupyter
#     # Por ejemplo:
#     # result_image_name = "miguel_08.jpg" # O cualquier otra imagen de resultados
#     # result_path = os.path.join(DIR_RESULTS, result_image_name)
#     # if os.path.exists(result_path):
#     #     result_img = cv2.cvtColor(cv2.imread(result_path), cv2.COLOR_BGR2RGB)
#     #     plt.figure(figsize=(10, 10))
#     #     plt.imshow(result_img)
#     #     plt.title(f"Resultado para {result_image_name}")
#     #     plt.axis('off')
#     #     plt.show()