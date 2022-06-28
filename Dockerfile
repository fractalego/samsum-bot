FROM nvcr.io/nvidia/tensorrt:22.04-py3
RUN mkdir /app
RUN mkdir /app/models
RUN mkdir /app/src
RUN mkdir /app/onnx

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
RUN apt update
RUN apt install -y openjdk-11-jdk
RUN apt install -y python3 python3-pip

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --upgrade requests

COPY ./compile.sh /app/compile.sh
COPY ./src/ /app/src/

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN pip install torch2trt/

COPY ./samsumbot_onnx/ onnx/

CMD ["sleep", "36000"]


#RUN python3 -m src.download_models

#RUN torch-model-archiver --model-name "bot" --version 0.0.1 \
#    --extra-files "/app/src/gptj_model.py,/app/models/gptj_model" \
#    --handler /app/src/handler.py --export-path /app/models/

#COPY config.properties /app/

#CMD ["torchserve", "--start", "--model-store", "models", "--models", "bot=bot.mar", "--foreground"]