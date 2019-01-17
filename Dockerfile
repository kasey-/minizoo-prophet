FROM python:3.6

COPY ./prophet /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "gunicorn", "--workers=1", "--timeout=120", "-b 0.0.0.0:80", "main:app" ]
