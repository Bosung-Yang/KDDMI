import os

os.system('sudo apt-get update ; sudo apt-get install nvidia-docker2 -y')
os.system('sudo systemctl restart docker')
os.system('sudo docker run -v ./data:/workspace/data -it --runtime=nvidia bosungyang/dmi:0809 bash')
