# Проект команды Cerebrum

Веб-приложение для интеллектуальной обработки данных о нарушениях безопасного проведения работ.

## Установка

Для установки и запуска приложения требуется установленный Python 3.10

1. Клонируйте репозиторий:

```bash
git clone https://github.com/mievst/DFO-2024.git
```

2. Установите зависимости:

```bash
cd ./DFO-2024
pip install -r requirements.txt
```

3. Скачайте веса по ссылке https://disk.yandex.ru/d/5DatvkC3gv464w и поместите
их в директорию DFO-2024\CerebrumFrontBack\checkpoints

## Использование

1. Перейдите в директорию проекта:

```bash
cd ./DFO-2024/CerebrumFrontBack
```

2. Запустите проект из директории проекта:

```bash
flask --app app run
```


2. Откройте приложение в браузере по адресу: `http://127.0.0.1:5000/`
