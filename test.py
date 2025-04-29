from tqdm import tqdm
from utils.parse_args_test import *
import scipy.io as scio
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
# Metric, loss .etc
from utils.utils import *
from utils.metric import *
from utils.loss import *
from utils.load_param_data import load_dataset, load_param
# my model
from models.MNHU_test import *
import numpy as np

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args = args
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(
                args.root, args.dataset, args.split_method)
        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size,
                                crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.test_data = DataLoader(
            dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False)
        # Network selection
        if  args.model == 'MNHU':
            model = MNHU()

        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

        # Load trained model
        checkpoint = torch.load(args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'],strict=False)

        # Test
        self.model.eval()
        self.mIoU.reset()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()

        with torch.no_grad():
            label_ = torch.Tensor().cuda()
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                if args.deep_supervision == 'DSV':
                    res = []
                    preds = self.model(data)
                    loss = 0

                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]

                elif args.deep_supervision == 'None':
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                losses.    update(loss.item(), pred.size(0))
                self.mIoU. update(pred, labels)

                _, mean_IOU = self.mIoU.get()

            print('miou', mean_IOU)

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    print('---------------------', args.model, '---------------------')
    main(args)
