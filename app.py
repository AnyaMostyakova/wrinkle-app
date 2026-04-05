from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from flask_cors import CORS
import uuid
import traceback
import time
import sys

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ============================================
# УКАЖИТЕ ПРАВИЛЬНЫЙ ПУТЬ К ВАШЕЙ МОДЕЛИ
# ============================================
MODEL_PATH = "model/wrinkle_model_best.pt"

print("="*60)
print("🚀 ЗАПУСК ПРИЛОЖЕНИЯ")
print("="*60)
print(f"📁 Текущая директория: {os.getcwd()}")
print(f"📁 Существует ли папка model?: {os.path.exists('model')}")
print(f"📁 Файл модели: {MODEL_PATH}")
print(f"📁 Существует ли модель?: {os.path.exists(MODEL_PATH)}")

# Проверяем содержимое папки model
if os.path.exists("model"):
    print(f"\n📁 Содержимое папки model/:")
    for f in os.listdir("model"):
        print(f"   - {f} ({os.path.getsize(os.path.join('model', f)) / 1024 / 1024:.2f} MB)")
else:
    print(f"\n⚠️ Папка model/ не существует!")
    print(f"   Создайте папку 'model' и поместите туда вашу модель")

print("\n" + "="*60)

from inference import WrinkleDetector

try:
    print("🔍 Загрузка детектора...")
    detector = WrinkleDetector(MODEL_PATH)
    print("✅ Детектор успешно загружен")
except Exception as e:
    print(f"❌ ОШИБКА загрузки детектора: {e}")
    traceback.print_exc()
    detector = None
    print("\n⚠️ Приложение будет работать, но анализ будет недоступен!")

print("="*60)


@app.route("/")
def index():
    print(f"📱 [GET] / - Запрос главной страницы")
    return render_template("index.html")


@app.route("/camera")
def camera():
    print(f"📱 [GET] /camera - Запрос страницы камеры")
    return render_template("camera.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()
    print("\n" + "="*60)
    print("📸 [POST] /analyze - ПОЛУЧЕН ЗАПРОС НА АНАЛИЗ")
    print("="*60)

    try:
        # Проверка детектора
        if detector is None:
            print("❌ Детектор не загружен!")
            return jsonify({"error": "Детектор не загружен. Проверьте модель"}), 500

        # Проверка файла
        if 'file' not in request.files:
            print("❌ Нет файла в запросе")
            return jsonify({"error": "Файл не загружен"}), 400

        file = request.files['file']
        print(f"📁 Получен файл: {file.filename}")

        if file.filename == '':
            print("❌ Пустое имя файла")
            return jsonify({"error": "Файл не выбран"}), 400

        # Чтение файла
        print("📖 Чтение файла...")
        file_bytes = file.read()
        print(f"📊 Размер файла: {len(file_bytes)} байт")

        if len(file_bytes) == 0:
            print("❌ Файл пуст")
            return jsonify({"error": "Пустой файл"}), 400

        # Декодирование изображения
        print("🖼️ Декодирование изображения...")
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Не удалось декодировать изображение")
            return jsonify({"error": "Не удалось прочитать изображение"}), 500

        print(f"✅ Изображение загружено: {img.shape} (H={img.shape[0]}, W={img.shape[1]}, C={img.shape[2]})")

        # Нормализация размера
        h, w = img.shape[:2]
        max_width = 800
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"📐 Изменение размера: {w}x{h} -> {new_w}x{new_h}")
            img = cv2.resize(img, (new_w, new_h))
        else:
            print(f"📐 Размер не изменён: {w}x{h}")

        # Конвертация в RGB
        print("🎨 Конвертация BGR -> RGB")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Проверка освещения
        print("\n💡 ПРОВЕРКА ОСВЕЩЕНИЯ:")
        is_lighting_good, lighting_score, lighting_message = detector.check_lighting_quality(img_rgb)
        print(f"   - Оценка: {lighting_score}%")
        print(f"   - Сообщение: {lighting_message}")
        print(f"   - Хорошее: {is_lighting_good}")

        # Коррекция освещения
        if not is_lighting_good:
            print("🔆 Выполняется коррекция освещения...")
            img_rgb = detector.auto_adjust_lighting(img_rgb)
            print("✅ Коррекция освещения выполнена")
        else:
            print("✅ Освещение хорошее, коррекция не требуется")

        # АНАЛИЗ МОРЩИН
        print("\n🔬 АНАЛИЗ МОРЩИН:")
        print("   - Вызов detector.predict()...")

        try:
            mask, binary, wrinkle_percent = detector.predict(img_rgb)
            print(f"   - Успешно! Процент морщин: {wrinkle_percent:.2f}%")
            print(f"   - Маска: min={mask.min():.3f}, max={mask.max():.3f}, среднее={mask.mean():.3f}")
            print(f"   - Бинарная маска: пикселей={binary.sum()}")
        except Exception as e:
            print(f"❌ Ошибка в predict: {e}")
            traceback.print_exc()
            raise

        # ВИЗУАЛИЗАЦИЯ
        print("\n🎨 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ:")
        try:
            result, skeleton = detector.create_training_like_visualization(img_rgb, mask, binary)
            print(f"   - Визуализация создана, скелет: {skeleton.sum()} пикселей")
        except Exception as e:
            print(f"❌ Ошибка визуализации: {e}")
            traceback.print_exc()
            raise

        # Добавление текста
        h, w = result.shape[:2]
        cv2.putText(result, f"Lighting: {lighting_score}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Wrinkles: {wrinkle_percent:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print("   - Текст добавлен")

        # СОХРАНЕНИЕ
        unique_id = str(uuid.uuid4())[:8]
        original_filename = f"original_{unique_id}.jpg"
        result_filename = f"result_{unique_id}.jpg"

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        print(f"\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
        print(f"   - Оригинал: {original_path}")
        cv2.imwrite(original_path, img)
        print(f"     ✅ Сохранён (размер: {os.path.getsize(original_path)} байт)")

        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, result_bgr)
        print(f"   - Результат: {result_path}")
        print(f"     ✅ Сохранён (размер: {os.path.getsize(result_path)} байт)")

        # ОТВЕТ
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Время выполнения: {elapsed_time:.2f} секунд")

        response = {
            "original": f"/uploads/{original_filename}",
            "result": f"/uploads/{result_filename}",
            "wrinkle_percent": float(wrinkle_percent),
            "lighting_score": lighting_score,
            "lighting_message": lighting_message
        }

        print(f"\n📤 ОТПРАВКА ОТВЕТА:")
        print(f"   - wrinkle_percent: {wrinkle_percent:.2f}%")
        print(f"   - lighting_score: {lighting_score}%")
        print(f"   - original: {response['original']}")
        print(f"   - result: {response['result']}")

        print("\n✅ АНАЛИЗ УСПЕШНО ЗАВЕРШЁН!")
        print("="*60 + "\n")

        return jsonify(response)

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА!")
        print(f"   - Время: {elapsed_time:.2f} секунд")
        print(f"   - Ошибка: {type(e).__name__}: {e}")
        print(f"\n📋 ПОЛНЫЙ СТЕК ОШИБКИ:")
        traceback.print_exc()
        print("\n" + "="*60 + "\n")

        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(f"📁 [GET] /uploads/{filename} - Запрос файла")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(404)
def not_found(error):
    print(f"⚠️ [404] Страница не найдена: {request.url}")
    return jsonify({"error": "Страница не найдена"}), 404


@app.errorhandler(500)
def internal_error(error):
    print(f"❌ [500] Внутренняя ошибка сервера: {error}")
    return jsonify({"error": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    import socket

    print("\n" + "="*60)
    print("🔍 ПОИСК СВОБОДНОГО ПОРТА")
    print("="*60)

    port = 5000
    for p in range(5000, 5010):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', p))
                port = p
                print(f"✅ Найден свободный порт: {port}")
                break
            except OSError:
                print(f"⚠️ Порт {p} занят, пробуем следующий...")
                continue

    print("\n" + "="*60)
    print(f"🚀 ЗАПУСК СЕРВЕРА")
    print("="*60)
    print(f"📱 Локальный доступ: http://127.0.0.1:{port}")
    print(f"📱 Сеть: http://192.168.0.108:{port}")
    print(f"🔧 Режим отладки: ВКЛЮЧЁН")
    print(f"📁 Папка загрузок: {UPLOAD_FOLDER}")
    print(f"📁 Абсолютный путь: {os.path.abspath(UPLOAD_FOLDER)}")
    print("="*60)
    print("\n👉 Нажмите CTRL+C для остановки сервера\n")

    app.run(host='0.0.0.0', port=port, debug=True)