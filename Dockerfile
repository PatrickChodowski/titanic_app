FROM python:3.7

ADD . /app
WORKDIR /app

RUN pip install -r ./app/requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.app:titanic_app"]