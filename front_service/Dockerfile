# Используйте официальный образ Python как базовый
FROM python:3.12-slim

# Установите рабочую директорию в контейнере
WORKDIR /app

# Установите необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    cmake build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируйте текущие файлы в контейнер
COPY . /app

# Установите зависимости из файла requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Откройте порт для связи с внешним миром (измените порт, если необходимо)
EXPOSE 5001

# Запустите приложение при запуске контейнера
CMD ["python", "app.py"]
