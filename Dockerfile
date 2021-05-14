FROM python:3.9

WORKDIR /tmp/dslr
COPY *.py ./
COPY requirements.txt ./
COPY datasets ./datasets

ENV DISPLAY=host.docker.internal:0

RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install -y zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

CMD [ "zsh" ]