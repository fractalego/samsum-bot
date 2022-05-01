FROM nvidia/cuda:11.
RUN mkdir /app
RUN mkdir /app/models
RUN mkdir /app/src

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
RUN apt update
RUN apt install -y nvidia-utils-470 nvidia-driver-470
RUN apt install -y openjdk-11-jdk
RUN apt install -y python3 python3-pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src/ /app/src/
RUN python3 -m src.download_model

RUN torch-model-archiver --model-name "bot" --version 0.0.1 \
    --extra-files "/app/src/gptj_model.py,/app/models/gptj_model" \
    --handler /app/src/handler.py --export-path /app/models/

CMD ["torchserve", "--start", "--model-store", "models", "--models", "bot=bot.mar", "--foreground"]
