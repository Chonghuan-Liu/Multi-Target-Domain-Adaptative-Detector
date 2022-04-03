from __future__ import  absolute_import
import os

import ipdb
import matplotlib
import torch
from tqdm import tqdm
import cv2 as cv
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=1000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    save_info = torch.load(opt.load_path_)
    faster_rcnn.load_state_dict(save_info['model'])
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in tqdm(range(opt.epoch)):
        #-----------------------------------------------------
        try:
            os.mkdir('./image_save/'+str(opt.FILEFOLDER))
        except:
            print("文件夹创建失败")
        #-----------------------------------------------------
        trainer.reset_meters()
        # -------------------21.11.5-------------------------------训练时多加一个参数 s_or_t 表示图片来自目标域还是原域
        for ii, (img, bbox_, label_, scale,img_target) in tqdm(enumerate(dataloader)):
            #-----------------------------------------------------------------
            opt.IMGINDEX=ii
            #----------------------------------------------------------------
            scale = at.scalar(scale)
            img, bbox, label,img_target = img.cuda().float(), bbox_.cuda(), label_.cuda(),img_target.cuda()
            trainer.train_step(img, bbox, label, scale,img_target,ii)
            #print(opt.plot_every)
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                # print(ori_img_.shape)
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                #-----------------------------------------------
                # save_info = torch.load(opt.save_path)
                # trainer.faster_rcnn.load_state_dict(save_info["model"])
                #-------------------------------------------------
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                # print("bbox:",_bboxes,"label:", _labels,
                #       "score:",_scores)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))

                trainer.vis.img('pred_img', pred_img)
                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
                #eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
                # # -------------------21.11.4--保存模型-----------------------------
                # save_info={
                #     "model":trainer.faster_rcnn.state_dict()
                # }
                # torch.save(save_info,opt.save_path)
                # # ---------------------------------------------------------------
        if(epoch%5==0):
            save_info={
                "model":trainer.faster_rcnn.state_dict()
            }
            torch.save(save_info,opt.save_path+str(epoch)+'.pth')

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 100:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        # if epoch == 13:
        #     break

        #----------------------------------------------------
        opt.FILEFOLDER+=1
        # ----------------------------------------------------


if __name__ == '__main__':
    import fire

    fire.Fire()
