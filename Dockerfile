FROM hub-dev.hexin.cn/jupyterhub/nvidia_cuda:py37-cuda100-ubuntu18.04-v2

COPY ./ /home/jovyan/task_oriented_dialogue_system 

RUN cd /home/jovyan/task_oriented_dialogue_system  && \
    python -m pip install -r requirements.txt 