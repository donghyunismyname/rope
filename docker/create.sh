NAME=rope
IMAGE=donghyunismyname/ubuntu2204:1
docker run -dit --name $NAME --gpus all -v /home:/home -v $HOME/.ssh:/root/.ssh -e "TERM=xterm-256color" $IMAGE zsh
