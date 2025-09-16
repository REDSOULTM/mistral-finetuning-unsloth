FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY RealizarFineTuning/mistral_finetuning_final.py ./
COPY Dataset_de_Miramar ./Dataset_de_Miramar

CMD ["python3", "mistral_finetuning_final.py"]
