FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN cp /usr/local/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda114.so /usr/local/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cpu.so

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]