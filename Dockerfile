FROM python:3.11.6-slim

RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "utils.py", "predict.py", "model.bin", "DictVectorizer.bin", "./"]

CMD [ "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000" ]