import os

os.system('aws s3 cp s3://bosung/G.tar ./')
os.system('aws s3 co s3://bosung/D.tar ./')
os.system('aws s3 cp s3://bosung/VGG16.tar')
os.system('aws s3 cp s3://bosung/VGG16_eval.tar')
os.system('aws s3 cp s3://bosung/VIB.tar')
os.system('aws s3 cp s3://bosung/VIB_eval.tar')
os.system('aws s3 cp s3://bosung/HSIC.tar')
os.system('aws s3 cp s3://bosung/HSIC_eval.tar')
