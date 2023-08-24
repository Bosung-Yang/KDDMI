import os

os.system('export MLFLOW_TRACKING_URI="http://218.155.110.67:5000"')
os.system('mlflow run ./ -P defense=VGG16 -P target=VGG16 --env-manager=local')
