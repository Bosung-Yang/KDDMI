import classify

def get_model(architecture,num_class):
    if architecture == 'linear':
        model = classify.VGG16(num_class)
    if architecture == 'softmax':
        model = classify.VGG16_VirtualSoftmax_1024(num_class)
    if architecture =='vgg16v':
        model = classify.VGG16_vanilla(1000)
    if architecture == 'vgg16':
        model = classify.VGG16(1000)
    elif architecture == 'vggvc':
        model = classify.VGG16(num_class)
    elif architecture == 'vgg16_softmax':
        model = classify.VGG16_Softmax(1000)
    elif architecture == 'vgg16_relu':
        model = classify.VGG16_ReLU(1000)
    elif architecture == 'vgg16_vs2000':
        model = classify.VGG16_VirtualSoftmax_2000(1000)
    elif architecture == 'vgg_sigmoid':
        model = classify.VGG16_Sigmoid(1000)
    elif architecture == 'vgg_virtualsoftmax_1024':
        model = classify.VGG16_VirtualSoftmax_1024(num_class)
    elif architecture == 'vgg_vs':
        model = classify.VGG16_VirtualSoftmax_1024(num_class)
    elif architecture == 'vgg16_vib':
        model = classify.VGG16_vib(1000)
    elif architecture=='vgg_vib_softmax':
        model = classify.VGG16_vib_softmax(1000)
    elif architecture=='resnet':
        model = classify.Resnet50(num_class)
    elif architecture == 'resnet_softmax':
        model = classify.Resnet50_softmax(num_class)
    elif architecture == 'resnet_vs':
        model = classify.IR152_vs1024(1024)
    elif architecture == 'resnet_vib':
        model = classify.IR152_vib(1000)
    elif architecture == 'facenet':
        model = classify.FaceNet64(1000)
    elif architecture == 'facenet_softmax':
        model = classify.FaceNet64_softmax(1000)
    elif architecture == 'facenet_vs':
        model = classify.FaceNet64_softmax(1024)
    elif architecture == 'facenet_vib':
        model = classify.FaceNet64_vib(1000)
    return model
