# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
import time
import wandb

np.set_printoptions(precision=4, suppress=True, threshold=10000)

from Network.VONet import VONet

class TartanVO(object):
    def __init__(self, model_name):
        # import ipdb;ipdb.set_trace()
        self.vonet = VONet()

        # load the whole model
        if model_name.endswith('.pkl'):
            modelname = 'models/' + model_name
            self.load_model(self.vonet, modelname)
            print('Model loaded... a7aaaaaaaaaa')

        self.vonet.cuda()

        self.test_count = 0
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
        self.flow_norm = 20 # scale factor for flow

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        print('Model loaded... a7aaaaaaaaaa number 2')
        return model

    def train_model(self, model, dataloader, optimizer, dataset_len, num_epochs = 10):
        # model = self.load_model(self.vonet, 'models/vo_model.pkl')
        # model = model
        print('Model Loaded')
        print('Start training...')
        running_loss = 0.0
        running_samples = 0
        for epoch in range(num_epochs):

            #To convert pose standard deviation to torch and moving into the GPU
            pose_std = torch.from_numpy(self.pose_std).cuda()
            model.train()
            #data length = 4
            for i, data in enumerate(dataloader):
                
                # if i%100 == 99:
                    # print('Epoch: {}, Iteration: {}'.format(epoch, i))
                # print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}/{dataset_len}')
                # get the inputs
                # print(len(data))
                # inputs, labels = data
                img0   = data['img1'].cuda()
                img1   = data['img2'].cuda()
                intrinsic = data['intrinsic'].cuda()
                motion = data['motion'].cuda()
                inputs = [img0, img1, intrinsic]
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    flow, pose = outputs
                    #or torch.mul(pose, pose_std)
                    pose = torch.div(pose, pose_std)
                    # pose = pose / pose_std # divide by the standard deviation
                    #
                    # pose_std = torch.from_numpy(self.pose_std).cuda()
                    # pose = pose * self.pose_std
                    # loss = criterion(outputs, labels)
                    #Now we have our own loss function (Up to Scale Loss Function)
                    loss = self.loss_function(GT=motion, Est=pose)
                    
                    loss.backward()
                    optimizer.step()
                    # optimizer.zero_grad()
                    # https://medium.com/deelvin-machine-learning/four-ways-to-increase-batch-size-in-deep-neural-network-training-a04ab3116088
                    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
                    #Gradient Accumulation:
                    # Imagine you want to use 32 images in one batch, but your hardware 
                    # crashes once you go beyond 8. In that case, you can use batches of 8 images and update weights once every 4 batches.
                    # if (i + 1) % 8 == 0 or i + 1 == len(dataloader):
                        # print('Gradient Accumulation')
                    # optimizer.step()  
                    # optimizer.zero_grad()
                    wandb.log({"Loss": loss})
                    running_loss += loss
                    running_samples += pose.shape[0]
                    # print(pose.shape[0]) #10
                    if i % 100 == 0:
                        print(f'ep: {epoch}, it: {i}, loss : {running_loss/running_samples:.5f}')
                        running_loss = 0.
                        running_samples = 0

                # print statistics
                # running_loss += loss.item()/dataset_len
                # wandb.log({"Running Loss": running_loss})
                # if (i+1) % 100 == 99:
                    # print('Epoch: {}, Iteration: {}'.format(epoch, i))
                    # print('Epoch:[%d], Iteration:[%5d] , running_loss: %.3f' % (epoch + 1, i + 1, running_loss))
            
            if epoch % 2 == 1:
                torch.save({
                    'model_state_dict': self.vonet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'./models/epoch_{epoch}_batch_{i}.pkl')
                print(f'Epoch: {epoch}, Model Saved')

        print('Finished Training')
        PATH = './models/vo_model_pretrained.pkl'
        # torch.save(self.vonet.state_dict(), PATH)
        torch.save(model.state_dict(), PATH)
        # torch.save(vonet.state_dict(), PATH)

    #Paper (TartanVO): https://arxiv.org/pdf/2011.00359.pdf
    #Up to Scale Loss Function
    def loss_function(self, GT, Est): #GT: ground truth, Est: estimated
        epsilon = torch.tensor(1e-6)
        # print('Ground Truth Shape', GT.shape)
        # print('Estimated Pose Shape:', Est.shape)
        # print(Est)
        trans_GT = GT[:, :3]
        rot_GT = GT[:, 3:]
        trans_Est = Est[:, :3]
        rot_Est = Est[:, 3:]
        trans_loss = torch.linalg.norm(trans_Est/torch.max(torch.linalg.norm(trans_Est), epsilon) - trans_GT/torch.max(torch.linalg.norm(trans_GT), epsilon))
        rot_loss = torch.linalg.norm(rot_Est-rot_GT)
        loss = trans_loss + rot_loss

        return loss
        

    def test_batch(self, sample):
        self.test_count += 1
        
        # import ipdb;ipdb.set_trace()
        img0   = sample['img1'].cuda()
        img1   = sample['img2'].cuda()
        intrinsic = sample['intrinsic'].cuda()
        inputs = [img0, img1, intrinsic]

        self.vonet.eval()

        with torch.no_grad():
            starttime = time.time()
            flow, pose = self.vonet(inputs)
            inferencetime = time.time()-starttime
            # import ipdb;ipdb.set_trace()
            posenp = pose.data.cpu().numpy()
            posenp = posenp * self.pose_std # The output is normalized during training, now scale it back
            flownp = flow.data.cpu().numpy()
            flownp = flownp * self.flow_norm

        # calculate scale from GT posefile
        if 'motion' in sample:
            motions_gt = sample['motion']
            scale = np.linalg.norm(motions_gt[:,:3], axis=1)
            trans_est = posenp[:,:3]
            trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
            posenp[:,:3] = trans_est 
        else:
            print('    scale is not given, using 1 as the default scale value..')

        print("{} Pose inference using {}s: \n{}".format(self.test_count, inferencetime, posenp))
        return posenp, flownp