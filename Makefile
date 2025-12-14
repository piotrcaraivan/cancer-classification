# Переменная: указываем, где лежит наш питон на Маке/Linux
VENV = .venv/bin/python

# 1. Установка библиотек
install:
	$(VENV) -m pip install -r requirements.txt

# 2. Обучение модели
train:
	$(VENV) cancer_prediction.py train

# 3. Оценка качества (Evaluate)
eval:
	$(VENV) cancer_prediction.py evaluate --model artifacts/model.joblib --split artifacts/split.joblib

# 4. Проверка стабильности (CV)
cv:
	$(VENV) cancer_prediction.py cv --folds 5

# 5. Прогноз для примера (для проверки)
predict:
	$(VENV) cancer_prediction.py predict --model artifacts/model.joblib --sample-index 0

# 6. Очистка (удалить старые модели, чтобы начать с нуля)
clean:
	rm -rf artifacts/*