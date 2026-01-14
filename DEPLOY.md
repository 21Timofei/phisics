# 🌐 Инструкция по деплою NoiseLab++

Руководство по развертыванию веб-интерфейса NoiseLab++ в production.

## Варианты деплоя

### 1. Streamlit Community Cloud (Рекомендуется) ✨

**Преимущества:**
- Бесплатно для публичных репозиториев
- Автоматический деплой из GitHub
- HTTPS из коробки
- Простота настройки

#### Шаги:

1. **Подготовка репозитория:**

```bash
# Убедитесь, что код на GitHub
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

2. **Создайте файл конфигурации:**

Создайте `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

Создайте `packages.txt` (если нужны системные зависимости):

```
# Системные пакеты (если требуются)
# libgomp1
```

3. **Деплой на Streamlit Cloud:**

   a. Зайдите на https://share.streamlit.io/

   b. Войдите через GitHub

   c. Нажмите "New app"

   d. Заполните форму:
      - Repository: `ваш-username/phisics`
      - Branch: `main`
      - Main file path: `web/streamlit_app.py`

   e. Нажмите "Deploy!"

4. **Ожидайте развертывания:**
   - Процесс занимает 5-10 минут
   - Вы получите URL вида: `https://ваш-username-phisics-streamlit-app.streamlit.app`

5. **Настройте секреты (если нужно):**
   - В настройках приложения на Streamlit Cloud
   - Добавьте переменные окружения через Secrets

---

### 2. Heroku 🚀

**Преимущества:**
- Поддержка больших приложений
- Простой CI/CD
- Возможность масштабирования

#### Шаги:

1. **Установите Heroku CLI:**

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Ubuntu
curl https://cli-assets.heroku.com/install.sh | sh

# Windows
# Скачайте установщик с https://devcenter.heroku.com/articles/heroku-cli
```

2. **Создайте файлы конфигурации:**

`Procfile`:
```
web: streamlit run web/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

`runtime.txt`:
```
python-3.11.7
```

`.slugignore` (опционально):
```
tests/
*.pdf
*.md
__pycache__/
*.pyc
.git/
```

3. **Деплой:**

```bash
# Логин в Heroku
heroku login

# Создайте приложение
heroku create noiselab-plus

# Задеплойте
git push heroku main

# Откройте в браузере
heroku open
```

4. **Настройка:**

```bash
# Масштабирование (если нужно больше ресурсов)
heroku ps:scale web=1

# Просмотр логов
heroku logs --tail

# Установка переменных окружения
heroku config:set VARIABLE_NAME=value
```

---

### 3. Docker + Cloud Run (Google Cloud) 🐳

**Преимущества:**
- Полный контроль
- Автоматическое масштабирование
- Pay-as-you-go

#### Шаги:

1. **Создайте Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Порт
EXPOSE 8080

# Запуск
CMD streamlit run web/streamlit_app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true
```

2. **Создайте `.dockerignore`:**

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.git/
.gitignore
*.md
tests/
.pytest_cache/
```

3. **Тестируйте локально:**

```bash
# Сборка образа
docker build -t noiselab-plus .

# Запуск контейнера
docker run -p 8080:8080 noiselab-plus
```

4. **Деплой на Google Cloud Run:**

```bash
# Установите Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Авторизация
gcloud auth login

# Настройка проекта
gcloud config set project your-project-id

# Включите Cloud Run API
gcloud services enable run.googleapis.com

# Деплой
gcloud run deploy noiselab-plus \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

---

### 4. AWS EC2 + Nginx 🖥️

**Преимущества:**
- Полный контроль над сервером
- Возможность использования больших ресурсов

#### Шаги:

1. **Создайте EC2 инстанс:**
   - Ubuntu 22.04 LTS
   - t3.medium или больше (рекомендуется для квантовых вычислений)
   - Security group: открыть порты 80, 443, 22

2. **Подключитесь к серверу:**

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Установите зависимости:**

```bash
# Обновите систему
sudo apt update && sudo apt upgrade -y

# Установите Python и pip
sudo apt install python3.11 python3.11-venv python3-pip nginx -y

# Клонируйте репозиторий
git clone https://github.com/your-username/phisics.git
cd phisics

# Создайте виртуальное окружение
python3.11 -m venv venv
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt
```

4. **Настройте systemd service:**

Создайте `/etc/systemd/system/noiselab.service`:

```ini
[Unit]
Description=NoiseLab++ Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/phisics
Environment="PATH=/home/ubuntu/phisics/venv/bin"
ExecStart=/home/ubuntu/phisics/venv/bin/streamlit run web/streamlit_app.py --server.port=8501 --server.address=localhost

Restart=always

[Install]
WantedBy=multi-user.target
```

5. **Настройте Nginx:**

Создайте `/etc/nginx/sites-available/noiselab`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_read_timeout 86400;
    }
}
```

6. **Активируйте и запустите:**

```bash
# Активируйте Nginx конфиг
sudo ln -s /etc/nginx/sites-available/noiselab /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Запустите сервис
sudo systemctl start noiselab
sudo systemctl enable noiselab

# Проверьте статус
sudo systemctl status noiselab
```

7. **Настройте SSL (опционально, но рекомендуется):**

```bash
# Установите Certbot
sudo apt install certbot python3-certbot-nginx -y

# Получите SSL сертификат
sudo certbot --nginx -d your-domain.com
```

---

### 5. Railway.app 🚂

**Преимущества:**
- Простой деплой из GitHub
- Бесплатный тарифный план
- Автоматический HTTPS

#### Шаги:

1. **Подготовьте `railway.json`:**

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run web/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

2. **Деплой:**
   - Зайдите на https://railway.app/
   - Подключите GitHub репозиторий
   - Выберите проект
   - Railway автоматически определит Python и задеплоит

---

## Общие рекомендации

### Оптимизация производительности

1. **Кэширование данных:**

Streamlit поддерживает встроенное кэширование:

```python
import streamlit as st

@st.cache_data
def expensive_computation():
    # Ваши вычисления
    pass
```

2. **Ограничение ресурсов:**

В `web/streamlit_app.py` добавьте ограничения:

```python
# Максимальное число кубитов
MAX_QUBITS = 2

# Максимальное число shots
MAX_SHOTS = 10000

# Максимальное число прогонов
MAX_RUNS = 50
```

### Мониторинг

1. **Логирование:**

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Tomography started")
```

2. **Метрики:**
   - Используйте Google Analytics для веб-метрик
   - Настройте Sentry для отслеживания ошибок

### Безопасность

1. **Переменные окружения:**

Используйте `.env` файл и `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')
```

2. **Rate limiting:**

Добавьте ограничение запросов:

```python
import time

if 'last_request' not in st.session_state:
    st.session_state.last_request = 0

if time.time() - st.session_state.last_request < 5:
    st.warning("Подождите 5 секунд между запросами")
else:
    # Выполняйте томографию
    st.session_state.last_request = time.time()
```

---

## Стоимость (примерные оценки)

| Платформа | Бесплатный тариф | Платный тариф |
|-----------|------------------|---------------|
| Streamlit Cloud | Публичные репо | $0 |
| Heroku | 550 часов/месяц | От $7/месяц |
| Google Cloud Run | $0 (до лимитов) | Pay-as-you-go (~$10-50/месяц) |
| AWS EC2 | t2.micro (750ч/мес) | От $10/месяц |
| Railway | $5 кредита/мес | От $5/месяц |

---

## Поддержка и обслуживание

### Обновление приложения

```bash
# Streamlit Cloud - просто push в GitHub
git push origin main

# Heroku
git push heroku main

# Docker/Cloud Run
docker build -t noiselab-plus . && docker push ...
gcloud run deploy ...

# EC2
ssh into server
cd phisics && git pull
sudo systemctl restart noiselab
```

### Резервное копирование

Регулярно делайте бэкапы:
- Код в Git
- Данные и логи
- Конфигурации

---

## Решение проблем

### Приложение не запускается

1. Проверьте логи
2. Убедитесь, что все зависимости установлены
3. Проверьте версию Python (требуется 3.11+)

### Медленная работа

1. Увеличьте ресурсы (CPU/RAM)
2. Используйте кэширование
3. Оптимизируйте вычисления

### Ошибки памяти

1. Ограничьте максимальное число кубитов
2. Уменьшите число shots
3. Используйте более мощный инстанс

---

## Контакты и поддержка

При возникновении проблем:
- Проверьте [документацию](README.md)
- Откройте issue на GitHub
- Проверьте логи приложения

Удачного деплоя! 🚀
