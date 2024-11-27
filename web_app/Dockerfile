# Используем базовый образ Python версии 3.11
FROM python:3.11-slim

# Устанавливаем переменную окружения для предотвращения создания файлов .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Устанавливаем переменную окружения для предотвращения буферизации вывода Python
ENV PYTHONUNBUFFERED=1

# Копируем все файлы текущей директории в директорию /app внутри контейнера
COPY . /app

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем зависимости из requirements.txt
# Используем --no-cache-dir для уменьшения размера конечного образа
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Указываем, что контейнер будет слушать порт 80
EXPOSE 80

# Создаем директорию .streamlit для конфигурации
RUN mkdir ~/.streamlit

# Копируем файлы конфигурации Streamlit в созданную директорию
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

# Указываем точку входа: запуск Streamlit
ENTRYPOINT ["streamlit", "run"]

# Указываем файл приложения (в данном случае main.py) для запуска
CMD ["main.py", "--server.port=80"]