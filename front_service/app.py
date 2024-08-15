from flask import Flask, render_template, Response
import cv2
import requests
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    
    """
    Отображает главную страницу.

    Эта функция возвращает HTML-шаблон `index.html`, который отображается
    при доступе к корневому URL-адресу приложения.

    Returns:
        str: Содержимое HTML-шаблона `index.html`.
    """
    
    return render_template('index.html')

def gen_frames():
    
    """
    Генерирует кадры с веб-камеры для потоковой передачи.

    Эта функция захватывает кадры с веб-камеры, отправляет их на сервер
    для обработки, получает обработанные кадры и возвращает их в формате
    JPEG для потоковой передачи.

    Yields:
        bytes: Обработанный кадр в формате JPEG, готовый к потоковой передаче.
    """
    
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Отправка кадра на сервер
            ret, buffer = cv2.imencode('.jpg', frame)
            response = requests.post('http://127.0.0.1:5002/process_frame', data=buffer.tobytes())

            # Получение обработанного кадра
            processed_frame = np.frombuffer(response.content, np.uint8)
            processed_frame = cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)

            # Перекодирование обработанного кадра для потоковой передачи
            ret, buffer = cv2.imencode('.jpg', processed_frame)

            # Возврат обработанного кадра
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    
    """
    Предоставляет потоковое видео с обработанными кадрами.

    Эта функция возвращает поток с обработанными кадрами в формате JPEG,
    полученными из функции `gen_frames`.

    Returns:
        Response: Потоковый ответ с кадрами в формате JPEG.
    """
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
