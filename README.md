# Model-Inversion-with-Deactivation

## Experimental setup
Our code is implemented using pytoch 1.4 and cudnn 10.1.
The required libararies can be installed by
```
pip install -r requirement.txt
```

## Dataset
You can download CelebA data in <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>.

## Train target model
To train vgg16 for face classification, you run
```
python train_vgg.py --save_path=./model.pth 
```
For adapting virtual softmax,
```
python train_vgg.py --student=vgg16_softmax --num_class=2000 --save_path=./model.pth 
```
For knowledge distillation,    
An exmple for the case Teahcer:virtual softmax, Student: linear,
```
python train_vgg.py --mode=kd --student=vgg16 --teacher=vgg16_softmax --teacher_path=./teacher.pth --temp=64 --num_class=2000 --save_path=./model.pth
```

## Model Inversion Attacks
To conduct GMI, you run
```
python attack_gmi.py --model=vgg16_softmax --path=./model.pth --exp=gmi --num_class=2000
```
For KEDMI,
```
python attack_ked.py --model=vgg16_softmax --path=./model.pth --exp=gmi --num_class=2000 --improved_flag --dist_flag
```
For DGMI, the target model should have nonlinear activation in the output layer.
```
python attack_dgmi.py --model=vgg16_softmax --path=./model.pth --exp=gmi --num_class=2000
```

## Reference
<https://github.com/SCccc21/Knowledge-Enriched-DMI>   
<https://github.com/MKariya1998/GMI-Attack>

## 0804 : edit inversion attack with mlflow
