import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from models.experimental import attempt_load
from models.yolo import Model
from utils.torch_utils import select_device

if __name__ == '__main__':
    print(torch.nn.Conv2d(1,2,1).f)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    # parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    # parser.add_argument('--rect', action='store_true', help='rectangular training')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    # parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    # parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    # parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    # parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    # parser.add_argument('--project', default='runs/train', help='save to project/name')
    # parser.add_argument('--name', default='exp', help='save to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # opt = parser.parse_args()
    # device = select_device('cpu', batch_size=16)
    # ckpt = torch.load('yolov5s.pt', map_location=device)  # load FP32 model
    # model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=4).to(device)
    #
    # dummy_input = torch.rand(16, 3, 640, 640)  # 假设输入20张1*28*28的图片
    # with SummaryWriter(comment='yolov5s') as w:
    #     w.add_graph(model, (dummy_input,))
