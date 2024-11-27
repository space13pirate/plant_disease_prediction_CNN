import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Получение текущей рабочей директории
working_dir = os.path.dirname(os.path.abspath(__file__))
# Указание пути к обученной модели
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.keras"
# Загрузка предварительно обученной модели
model = tf.keras.models.load_model(model_path)
# Загрузка индексов классов (словарь индексов и их соответствующих названий)
# Загрузка JSON-файла с указанием кодировки UTF-8
with open(f"{working_dir}/class_indices_rus.json", 'r', encoding='utf-8') as file:
    class_indices = json.load(file)

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path, target_size=(150, 150)):
    # Загрузка изображения с помощью Pillow
    img = Image.open(image_path)
    # Изменение размера изображения до заданных размеров
    img = img.resize(target_size)
    # Преобразование изображения в массив numpy
    img_array = np.array(img)
    # Добавление измерения "batch" для подачи в модель (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    # Нормализация значений пикселей в диапазон [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Функция для предсказания класса изображения
def predict_image_class(model, image_path, class_indices):
    # Предобработка изображения (загрузка, изменение размера, нормализация)
    preprocessed_img = load_and_preprocess_image(image_path)
    # Выполнение предсказания с использованием модели
    predictions = model.predict(preprocessed_img)
    # Определение индекса класса с наибольшей вероятностью
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Поиск названия класса по индексу
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Название веб-приложения
st.markdown("<h1 style='text-align: center;'>🌿 Распознавание болезней растений</h1>", unsafe_allow_html=True)

# Поле для загрузки изображения
uploaded_image = st.file_uploader("Загрузите изображение...", type=["jpg", "jpeg", "png"])

# Проверяем, было ли загружено изображение
if uploaded_image is not None:
    # Открываем изображение с помощью Pillow
    image = Image.open(uploaded_image)

    # Создаём два столбца для отображения интерфейса
    col1, col2 = st.columns(2)

    with col1:
        # Изменяем размер изображения до 150x150 для отображения
        resized_img = image.resize((150, 150))
        # Отображаем изображение в первом столбце
        st.image(resized_img)

    with col2:
        # Добавляем кнопку для классификации изображения
        if st.button('Определить'):
            # Предобработка изображения и получение предсказания
            prediction = predict_image_class(model, uploaded_image, class_indices)
            # Отображение результата предсказания
            st.success(f'Результат: {str(prediction)}')