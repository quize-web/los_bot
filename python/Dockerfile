# обязательно версию 3.6, на более новых проблема с tensorflow
FROM python:3.6

WORKDIR /usr/src/app

# установка базовых утилит
RUN apt-get update && apt-get install -y \
    	git \
    	gcc \
    	libglib2.0 \
		libsm6 \
		libfontconfig1 libxrender1 \
    	libxtst6 \
    	ffmpeg \
        libxext6 \
        libgl1-mesa-glx

# клонируем репозиторий
RUN mkdir src \
    && cd src \
    && git clone https://github.com/ria-com/nomeroff-net.git \
    && cd nomeroff-net \
    && git clone https://github.com/youngwanLEE/centermask2.git \
    && ls /usr/src/app/src/nomeroff-net -la

# установка python-пакетов
RUN pip install opencv-python
RUN pip install torch==1.7.1
RUN pip3 install PyYAML==5.3
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip3 install torchvision==0.8.2
RUN pip3 install Cython
RUN pip3 install numpy

# установка python-пакетов из requirements.txt
RUN pip3 install -r ./src/nomeroff-net/requirements.txt

CMD [ "python", "./src/nomeroff-net/run_los.py" ]