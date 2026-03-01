FROM pytorch/pytorch:1.8.1-cpu-py3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir .

CMD ["python"]

