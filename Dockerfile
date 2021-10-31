FROM redhat/ubi8:8.4-211
COPY . /app
WORKDIR /app
RUN yum install -y python38 && \
    yum install -y mesa-libGL && \
    pip3 install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD python3 app.py
