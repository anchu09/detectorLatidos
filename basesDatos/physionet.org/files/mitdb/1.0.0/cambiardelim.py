import os

# Función para reemplazar "\," con ","
def replace_comma(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(content.replace('\\,', ','))

# Carpeta principal
folder_path = '/home/dani/tfg/detectorLatidos/basesDatos/physionet.org/files/mitdb/1.0.0/senales_troceadas/anotaciones'

# Recorrer todas las subcarpetas y leer todos los archivos en ellas
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Ignorar archivos que no tienen la extensión .txt
        if file.endswith('.csv'):
            filepath = os.path.join(root, file)
            replace_comma(filepath)