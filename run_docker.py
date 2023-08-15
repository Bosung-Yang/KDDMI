import os

os.system('sudo apt-get update ; sudo apt-get install nvidia-docker2 -y')
os.system('sudo systemctl restart docker')
os.system('sudo docker run -v ./:/workspace/KDDMI -it --shm-size=8G --runtime=nvidia bosungyang/dmi:data bash')
