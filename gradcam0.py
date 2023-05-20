import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# from torchvision.models.segmentation import deeplabv3_resnet50
from nets.deeplabv3_plus import DeepLab
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.grad_cam import GradCAM


# image = np.array(Image.open('D:/Semantic segmentation/Deeplabv3+/VOCdevkit/JPEGImages/test/507.jpg'))
image = np.array(Image.open('C:/Users/user/Desktop/587.jpg'))
rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = DeepLab()
model = model.eval()
#模型权重加载
weight_pth = 'D:/Semantic segmentation/Deeplabv3+/logs/mobilenetv2/2ceng/best_epoch_weights.pth'
model.load_state_dict(torch.load(weight_pth), strict = False)

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()
output = model(input_tensor)

normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
sem_classes = ['background', 'landslide', 'water', 'load', 'sky', 'vegetation']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

landslide_category = sem_class_to_idx["landslide"]
landslide_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
landslide_mask_uint8 = 255 * np.uint8(landslide_mask == landslide_category)
landslide_mask_float = np.float32(landslide_mask == landslide_category)

# vegetation_category = sem_class_to_idx["vegetation"]
# vegetation_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
# vegetation_mask_uint8 = 255 * np.uint8(vegetation_mask == vegetation_category)
# vegetation_mask_float = np.float32(vegetation_mask == vegetation_category)

# water_category = sem_class_to_idx["water"]
# water_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
# water_mask_uint8 = 255 * np.uint8(water_mask == water_category)
# water_mask_float = np.float32(water_mask == water_category)

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

target_layers = [model.cls_conv]
targets = [SemanticSegmentationTarget(landslide_category, landslide_mask_float)]
# targets = [SemanticSegmentationTarget(water_category, water_mask_float)]
# targets = [SemanticSegmentationTarget(vegetation_category, vegetation_mask_float)]

with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor = input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

img = Image.fromarray(cam_image)
img.show()
img.save('D:/Semantic segmentation/Deeplabv3+/VOCdevkit/cam/587.png')