import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize


class WrinkleDetector:

    def __init__(self, model_path):
        print(f"🔍 Загрузка модели из: {model_path}")
        self.model = YOLO(model_path)
        print(f"✅ YOLO модель загружена")

    def predict(self, image):
        """
        Анализ изображения как в Colab коде
        """
        # Сохраняем оригинальный размер
        h, w = image.shape[:2]

        # Предсказание с низким порогом (как в Colab)
        results = self.model(image, conf=0.05, verbose=False)

        # Создаем маску (как в Colab)
        mask = np.zeros((h, w), dtype=np.float32)

        # Получаем маски из результатов
        if results[0].masks is not None:
            # Для YOLO версии 8.x
            if hasattr(results[0].masks, 'data'):
                masks = results[0].masks.data.cpu().numpy()
                for m in masks:
                    m_resized = cv2.resize(m, (w, h))
                    mask = np.maximum(mask, m_resized)
            else:
                # Альтернативный способ
                for m in results[0].masks:
                    m_np = m.cpu().numpy()
                    m_resized = cv2.resize(m_np, (w, h))
                    mask = np.maximum(mask, m_resized)

        # Бинаризация (как в Colab: mask > 0.4)
        binary = (mask > 0.4).astype(np.uint8) * 255

        # Процент морщин
        if binary.size > 0:
            wrinkle_percent = (binary.sum() / (binary.size * 255)) * 100
        else:
            wrinkle_percent = 0.0

        return mask, binary, wrinkle_percent

    def skeletonize_mask(self, binary):
        """Скелетизация как в Colab коде"""
        if binary is None or binary.sum() == 0:
            return np.zeros_like(binary) if binary is not None else np.zeros((100, 100))

        # Скелетизация
        skeleton = skeletonize(binary > 0).astype(np.uint8) * 255

        # Удаляем слишком маленькие объекты
        num_labels, labels = cv2.connectedComponents(skeleton)
        for label_id in range(1, num_labels):
            if np.sum(labels == label_id) < 15:
                skeleton[labels == label_id] = 0

        return skeleton

    def create_training_like_visualization(self, image, mask, binary):
        """Визуализация как в Colab коде - красные линии поверх фото"""
        # Получаем скелет (тонкие линии)
        skeleton = self.skeletonize_mask(binary)

        # Создаем overlay как в Colab: красные линии
        overlay = image.copy()

        if skeleton is not None and skeleton.sum() > 0:
            # Красные линии (как в Colab: [255, 0, 0])
            overlay[skeleton == 255] = [255, 0, 0]

        return overlay, skeleton if skeleton is not None else np.zeros_like(binary)

    def check_lighting_quality(self, image):
        """Проверяет качество освещения"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        score = 0

        if 100 <= mean_brightness <= 150:
            score += 40
        elif 80 <= mean_brightness <= 170:
            score += 30
        elif 60 <= mean_brightness <= 190:
            score += 20
        else:
            score += 10

        if 40 <= std_brightness <= 80:
            score += 30
        elif 30 <= std_brightness <= 90:
            score += 20
        else:
            score += 10

        overexposed = np.sum(gray > 250) / gray.size * 100
        underexposed = np.sum(gray < 5) / gray.size * 100

        if overexposed < 1:
            score += 20
        elif overexposed < 5:
            score += 10

        if underexposed < 1:
            score += 10
        elif underexposed < 3:
            score += 5

        score = min(100, score)
        is_good = score >= 60

        if score >= 80:
            message = "Отличное освещение!"
        elif score >= 60:
            message = "Освещение приемлемое"
        elif score >= 40:
            message = "Освещение плохое. Включите больше света"
        else:
            message = "Освещение очень плохое. Включите яркий свет"

        return is_good, score, message

    def auto_adjust_lighting(self, image):
        """Автоматическая коррекция освещения"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return result