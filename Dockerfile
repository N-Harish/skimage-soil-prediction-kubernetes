FROM ubuntu
COPY . /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y libsm6 libxrender1 libfontconfig1 libice6 libgl1-mesa-glx && \
    apt-get -y install libglib2.0-0  && \
    apt install -y python3.8 && \
    apt install -y python3-pip && \
    pip3 install -r requirements.txt
EXPOSE 5000
CMD python3 app.py