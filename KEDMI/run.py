import os

#os.system('mlflow run ./ -P defense=VGG16 -P target=VGG16 --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=VGG16 -P target=VIB --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=VGG16 -P target=HSIC --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=VGG16 -P target=KD --env-manager=local --experiment-name=kedmi')

#os.system('mlflow run ./ -P defense=VIB -P target=VGG16 --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=VIB -P target=VIB --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=VIB -P target=HSIC --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=VIB -P target=KD --env-manager=local --experiment-name=kedmi')

#os.system('mlflow run ./ -P defense=HSIC -P target=VGG16 --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=HSIC -P target=VIB --env-manager=local --experiment-name=kedmi')
os.system('mlflow run ./ -P defense=HSIC -P target=HSIC --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=HSIC -P target=KD --env-manager=local --experiment-name=kedmi')

#os.system('mlflow run ./ -P defense=KD -P target=VGG16 --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=KD -P target=VIB --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=KD -P target=HSIC --env-manager=local --experiment-name=kedmi')
#os.system('mlflow run ./ -P defense=KD -P target=KD --env-manager=local --experiment-name=kedmi')
