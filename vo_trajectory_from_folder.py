##Pytorch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
##Datasets
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

#wandb
import wandb

def get_args():
    parser = argparse.ArgumentParser(description='HRL')
    
    #Choose train or test
    parser.add_argument('--train_test', type=str, default='test', help='train or test')
    #Batch size
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    #Number of epochs
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs (default: 10)')
    #Number of Workers
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    #Image Width
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    #Image Height
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    #Model Name
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    #Choose Unity
    parser.add_argument('--unity', action='store_true', default=False,
                        help='unity test (default: False)') 

    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')

    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    
    #Choose optical flow
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    #load wandb
    wandb.init(project="tartan_vo", entity="ahmedharbi")

    # load trajectory data from a folder
    # datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    elif args.unity:
        datastr = 'unity'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])


    #loading model if train or test
    #Test Mode
    if args.train_test == 'test':
        print('Test Mode Selected')
        testvo = TartanVO(args.model_name)
        testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
        testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
        testDataiter = iter(testDataloader)
    ## Train Mode
    else: 
        trainvo = TartanVO(args.model_name)
        print('Train Mode Selected')
        trainDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
        trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
        # trainDataiter = iter(trainDataloader)
        
        lr = 0.0001
        decay = 0.1
        config = wandb.config
        wandb.log({"lr": lr, "decay": decay})
        #vonet is our network
        optimizer = optim.SGD(trainvo.vonet.parameters(), lr=lr, weight_decay=decay)
        criterion = nn.MSELoss()
        trainvo.train_model(dataloader=trainDataloader, optimizer=optimizer, num_epochs=args.epochs, dataset_len=len(trainDataset))


    
    if args.train_test == 'test':
        motionlist = []
        testname = datastr + '_' + args.model_name.split('.')[0]
        if args.save_flow:
            flowdir = 'results/'+testname+'_flow'
            if not isdir(flowdir):
                mkdir(flowdir)
            flowcount = 0
        while True:
            try:
                sample = testDataiter.next()
            except StopIteration:
                break

            motions, flow = testvo.test_batch(sample)
            motionlist.extend(motions)

            if args.save_flow:
                for k in range(flow.shape[0]):
                    flowk = flow[k].transpose(1,2,0)
                    np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                    flow_vis = visflow(flowk)
                    cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                    flowcount += 1

        poselist = ses2poses_quat(np.array(motionlist))

        # calculate ATE, RPE, KITTI-RPE
        if args.pose_file.endswith('.txt'):
            evaluator = TartanAirEvaluator()
            results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
            if datastr=='euroc':
                print("==> ATE: %.4f" %(results['ate_score']))
            else:
                print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

            # save results and visualization
            plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
            np.savetxt('results/'+testname+'.txt',results['est_aligned'])
        else:
            np.savetxt('results/'+testname+'.txt',poselist)