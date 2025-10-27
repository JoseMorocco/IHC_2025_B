import cv2
import os

def detectar_y_guardar_rostros(input_folder, output_folder, classifier_path):
    # Cargar el clasificador pre-entrenado para rostros
    face_cascade = cv2.CascadeClassifier(classifier_path)

    if face_cascade.empty():
        print(f"Error: No se pudo cargar el clasificador desde {classifier_path}. Asegúrate de que el archivo XML existe y la ruta es correcta.")
        return

    # Asegurarse de que la carpeta de resultados existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Carpeta de resultados creada: {output_folder}")

    # Contador para las imágenes procesadas
    imagenes_procesadas = 0
    rostros_totales_detectados = 0

    # Iterar sobre cada archivo en la carpeta de entrada
    for filename in os.listdir(input_folder):
        # Filtrar solo archivos de imagen (puedes añadir más extensiones si es necesario)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            print(f"Procesando imagen: {image_path}")

            # Leer la imagen
            img = cv2.imread(image_path)

            if img is None:
                print(f"Advertencia: No se pudo leer la imagen {filename}. Puede que esté corrupta o el formato no sea compatible.")
                continue

            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Realizar la detección de rostros
            # Ajusta scaleFactor y minNeighbors según la calidad de tus imágenes
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Dibujar rectángulos alrededor de los rostros detectados
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # Color verde, grosor 2
            
            # Construir la ruta de salida para la imagen procesada
            output_image_path = os.path.join(output_folder, f"detectado_{filename}")

            # Guardar la imagen con los rostros detectados en la carpeta de resultados
            cv2.imwrite(output_image_path, img)
            print(f"Guardado: {output_image_path} ({len(faces)} rostros detectados)")
            
            imagenes_procesadas += 1
            rostros_totales_detectados += len(faces)

    print(f"\n--- Proceso Finalizado ---")
    print(f"Imágenes procesadas: {imagenes_procesadas}")
    print(f"Rostros totales detectados: {rostros_totales_detectados}")
    print(f"Resultados guardados en: {output_folder}")

if __name__ == "__main__":
    # --- Configuración de rutas ---
    # La carpeta 'imgs' dentro de tu directorio PROPUESTO
    input_images_folder = 'imgs' 
    
    # La carpeta 'resultados' dentro de tu directorio PROPUESTO
    output_results_folder = 'resultados' 
    
    # El clasificador Haar Cascade que debes descargar en la misma carpeta que este script
    haar_cascade_classifier = 'haarcascade_frontalface_default.xml' 

    # Llamar a la función principal
    detectar_y_guardar_rostros(input_images_folder, output_results_folder, haar_cascade_classifier)