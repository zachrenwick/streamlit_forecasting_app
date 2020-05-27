FROM python:3.8

EXPOSE 8501

COPY requirements.txt .
COPY app.py .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run"]
CMD ["app.py"]