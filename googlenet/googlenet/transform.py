from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize

# transform = ToTensor()

transform = Compose([
    Resize(256),               # 将输入图片大小调整为256x256
    CenterCrop(224),           # 从图片中心裁剪出224x224的区域
    ToTensor(),                # 将图片转换为Tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准化
])