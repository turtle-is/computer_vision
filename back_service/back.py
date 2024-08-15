from flask import Flask, request, jsonify
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io

app = Flask(__name__)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    
    """
    Обрабатывает входной кадр с изображением.

    Эта функция получает изображение в формате байтов через POST-запрос,
    декодирует его в формате OpenCV, выполняет обнаружение лиц с помощью
    библиотеки `face_recognition`, рисует прямоугольники вокруг обнаруженных
    лиц, а затем возвращает обработанное изображение в формате JPEG.

    Returns:
        bytes: Обработанное изображение в формате JPEG.
    """
    
    # Преобразование входного изображения из байтов
    npimg = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Обнаружение лиц
    face_locations = face_recognition.face_locations(img)
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Рисование прямоугольников вокруг лиц
    for (top, right, bottom, left) in face_locations:
        draw.rectangle([(left, top), (right, bottom)], outline="yellow", width=5)

    # Преобразование обратно в формат OpenCV и кодирование как JPEG
    del draw
    processed_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', processed_img)
    
    return buffer.tobytes()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
