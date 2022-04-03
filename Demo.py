from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize,preprocess,TransformTest
from model import FasterRCNNVGG16
from utils import array_tool as at
from utils.vis_tool import visdom_bbox,vis_bbox,fig4vis
from data.util import read_image
import torch
from domain_transfer_network.network import TransformNetwork
from matplotlib import pyplot as plot


def visdom_bbox_test(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    return fig
def load_transform_network():
    transform_network = TransformNetwork()
    transform_network.load_state_dict(torch.load('./domain_transfer_network/model/vgg19_30_30_4_8500.pth'))
    return transform_network



faster_rcnn=FasterRCNNVGG16()
faster_rcnn.cuda()
save_info=torch.load(opt.load_path_)
faster_rcnn.load_state_dict(save_info['model'])
transform_network = load_transform_network()
transform_network.cuda()


img=read_image('./demo_img/input/12_bright_1.jpg')
tsf=TransformTest(opt.min_size, opt.max_size)
img=tsf(img)

img=inverse_normalize(at.tonumpy(img))
# size=img.shape[1:]
# img_ = preprocess(at.tonumpy(img))
# img_=torch.Tensor(img_).unsqueeze(0)
pred_bboxes,pred_labels,pred_scores=faster_rcnn.predict([img],visualize=True,transfer_model=transform_network)
path='./demo_img/output/12_bright_1.jpg'

fig =   vis_bbox(img,
       at.tonumpy(pred_bboxes[0]),
       at.tonumpy(pred_labels[0]).reshape(-1),
       at.tonumpy(pred_scores[0]))
fig_ = fig.get_figure()
fig_.savefig(path)
plot.close()
