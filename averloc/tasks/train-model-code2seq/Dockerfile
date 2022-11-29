FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN echo "deb http://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
RUN echo "deb http://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get install -y nvidia-modprobe

RUN pip3 install requests rouge tqdm

#RUN nvidia-smi

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY models/code2seq-merged /code2seq
COPY task/ /app/

ENTRYPOINT [ "/app/entrypoint.sh" ]
