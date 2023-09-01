import os

os.system('mlflow run ./ -P defense=VGG16 -P target=KD --env-manager=local --experiment-name=gmi-t-bido1')
os.system('mlflow run ./ -P defense=KD -P target=KD --env-manager=local --experiment-name=gmi-t-bido1')
#os.system('mlflow run ./ -P defense=VIB -P target=KD --env-manager=local --experiment-name=gmi')
#os.system('mlflow run ./ -P defense=HSIC -P target=KD --env-manager=local --experiment-name=gmi')
#os.system('mlflow run ./ -P defense=VIB -P target=VIB --env-manager=local --experiment-name=gmi')
os.system('mlflow run ./ -P defense=KD -P target=HSIC --env-manager=local --experiment-name=gmi-t-bido1')
os.system('mlflow run ./ -P defense=KD -P target=VGG16 --env-manager=local --experiment-name=gmi-t-bido1')
os.system('mlflow run ./ -P defense=KD -P target=VIB --env-manager=local --experiment-name=gmi-t-bido1')

