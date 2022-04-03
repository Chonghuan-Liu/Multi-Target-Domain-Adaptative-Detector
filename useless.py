import torchvision
import torch
img=torch.ones(size=(100,100))
img*=0.5
print(img)
img=img.squeeze(0)
torchvision.utils.save_image(img, './mask_imgs/' + 'test' + '.jpg', padding=0)