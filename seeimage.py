import csv
import base64
import numpy as np
import cv2
import os

# Aumentar el límite del tamaño del campo CSV
csv.field_size_limit(100000000)

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def save_image_from_base64(image_str, image_folder):
    # Decodificar la cadena base64 en una matriz numpy
    decoded_data = base64.b64decode(image_str)
    np_data = np.frombuffer(decoded_data, np.uint8)
    
    # Decodificar la matriz numpy en una imagen
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    
    # Generar un nombre de archivo único para la imagen
    image_name = f"image_{hash(image_str)}.jpg"
    
    # Guardar la imagen en disco
    image_path = os.path.join(image_folder, image_name)
    cv2.imwrite(image_path, img)
    
    return image_name

if __name__ == '__main__':
    CSV_FILE = "info/speed_data.csv"
    IMAGE_FOLDER = "images"
    
    try:
        csv_data = read_csv_file(CSV_FILE)
        print("Vehicle Data:")
        for row in csv_data:
            print(f"Vehicle Name: {row['Vehicle Name']}, Vehicle ID: {row['Vehicle ID']}, Speed (Km/h): {row['Speed (Km/h)']}, Image Path: {row['Image']}")
            image_name = save_image_from_base64(row['Image'], IMAGE_FOLDER)
            # Actualizar el nombre de la imagen en el CSV
            row['Image'] = image_name
    
        # Guardar los datos actualizados en el CSV
        with open(CSV_FILE, 'w', newline='') as file:
            fieldnames = ['Vehicle Name', 'Vehicle ID', 'Speed (Km/h)', 'Image']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
            
    except FileNotFoundError:
        print(f"El archivo CSV '{CSV_FILE}' no fue encontrado.")
