FROM python:3.6-slim
WORKDIR /song_predict2
COPY requirements.txt /song_predict2/requirements.txt
RUN pip install -r requirements.txt

COPY . /song_predict2
ENTRYPOINT [ "python" ]
CMD ["WebUI.py"]
