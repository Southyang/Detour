import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from model.utils import log_normal_density


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 1, -1)

class DetourPolicy_nodiffusion(nn.Module):
    def __init__(self, frames, action_space):
        super(DetourPolicy_nodiffusion, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))
        self.obs_encoding_size = 20

        # laser
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 20)

        # camera
        self.act_fea_cv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8 ,stride=4)
        self.act_fea_cv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.act_fea_cv5 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1)
        # self.act_fc3 = nn.Linear(43776,20)
        self.act_fc3 = nn.Linear(768,20)

        # other
        self.act_fc4 = nn.Linear(2+2+4, 20)

        # concat laser camera and other
        self.act_fc2 = nn.Linear(20+20+20, 30)

        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=2,
            dim_feedforward=2*self.obs_encoding_size,
            activation="gelu"
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=2)

        # output
        self.actor1 = nn.Linear(30, 1)
        self.actor2 = nn.Linear(30, 1)

        # laser
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 20)

        # camera 3 45 80
        self.crt_fea_cv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8 ,stride=4)
        self.crt_fea_cv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.crt_fea_cv5 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1)
        # self.crt_fc3 = nn.Linear(43776,20) # 320 180
        self.crt_fc3 = nn.Linear(768,20) # 80 45

        # other
        self.crt_fc4 = nn.Linear(2+2+4, 20)

        # concat laser camera and other
        self.crt_fc2 = nn.Linear(20+20+20, 30)

        # output
        self.critic = nn.Linear(30, 1)

    def forward(self, x, img, goal, speed, orientation):
        # action
        # laser
        a = F.relu(self.act_fea_cv1(x)) # 1 32 255
        a = F.relu(self.act_fea_cv2(a)) # 1 32 128
        a = a.view(a.shape[0], -1) # 1 4096
        a = F.relu(self.act_fc1(a)).unsqueeze(1) # 1 20

        # camera
        # img: np 1 3 80 45
        a2 = F.relu(self.act_fea_cv3(img)) # 1 32 19 10
        a2 = F.relu(self.act_fea_cv4(a2)) # 1 64 8 4
        a2 = F.relu(self.act_fea_cv5(a2)) # 1 64 6 2
        a2 = a2.reshape(a2.shape[0], -1) # 1 768
        a2 = F.relu(self.act_fc3(a2)).unsqueeze(1) # 1 20

        # other
        a3 = torch.cat((goal, speed, orientation), dim=-1)
        a3 = F.relu(self.act_fc4(a3)).unsqueeze(1) # 1 20

        # concat
        a4 = torch.cat((a, a2, a3), dim=1) # 1 60
        a4 = self.sa_encoder(a4)
        a4 = a4.flatten(start_dim=1)
        a4 = F.relu(self.act_fc2(a4))
        mean1 = F.sigmoid(self.actor1(a4))
        mean2 = F.tanh(self.actor2(a4))
        mean = torch.cat((mean1, mean2), dim=-1)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        # laser
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v)).unsqueeze(1)

        # camera
        v2 = F.relu(self.crt_fea_cv3(img))
        v2 = F.relu(self.crt_fea_cv4(v2))
        v2 = F.relu(self.act_fea_cv5(v2))
        v2 = v2.reshape(v2.shape[0], -1)
        v2 = F.relu(self.crt_fc3(v2)).unsqueeze(1)

        # other
        v3 = torch.cat((goal, speed, orientation), dim=-1)
        v3 = F.relu(self.crt_fc4(v3)).unsqueeze(1)

        # concat
        v4 = torch.cat((v, v2, v3), dim=1)
        v4 = self.sa_encoder(v4)
        v4 = v4.flatten(start_dim=1)
        v4 = F.relu(self.crt_fc2(v4))
        v4 = self.critic(v4)

        return v4, action, logprob, mean

    def evaluate_actions(self, x, img, goal, speed, orientation, action):
        v, _, _, mean = self.forward(x, img, goal, speed, orientation)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

class DetourPolicy(nn.Module):
    def __init__(self, frames, action_space):
        super(DetourPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))
        self.obs_encoding_size = 20
        self.alpha = 0.1

        # laser
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 20)

        # camera
        self.act_fea_cv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8 ,stride=4)
        self.act_fea_cv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.act_fea_cv5 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1)
        # self.act_fc3 = nn.Linear(43776,20)
        self.act_fc3 = nn.Linear(768,20)

        # other
        self.act_fc4 = nn.Linear(2+2+4, 20)

        # concat laser camera and other
        self.act_fc2 = nn.Linear(20+20+20, 30)

        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=2,
            dim_feedforward=2*self.obs_encoding_size,
            activation="gelu"
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=2)

        # output
        self.actor1 = nn.Linear(30, 1)
        self.actor2 = nn.Linear(30, 1)

        # laser
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 20)

        # camera 3 45 80
        self.crt_fea_cv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8 ,stride=4)
        self.crt_fea_cv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.crt_fea_cv5 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3,stride=1)
        # self.crt_fc3 = nn.Linear(43776,20) # 320 180
        self.crt_fc3 = nn.Linear(768,20) # 80 45

        # other
        self.crt_fc4 = nn.Linear(2+2+4, 20)

        # concat laser camera and other
        self.crt_fc2 = nn.Linear(20+20+20, 30)

        # output
        self.critic = nn.Linear(30, 1)

    def forward(self, x, img, img_pre, goal, speed, orientation):
        # action
        # laser
        a = F.relu(self.act_fea_cv1(x)) # 1 32 255
        a = F.relu(self.act_fea_cv2(a)) # 1 32 128
        a = a.view(a.shape[0], -1) # 1 4096
        a = F.relu(self.act_fc1(a)).unsqueeze(1) # 1 20

        # camera
        # img: np 1 3 80 45
        a2 = F.relu(self.act_fea_cv3(img)) # 1 32 19 10
        a2 = F.relu(self.act_fea_cv4(a2)) # 1 64 8 4
        a2 = F.relu(self.act_fea_cv5(a2)) # 1 64 6 2
        a2 = a2.reshape(a2.shape[0], -1) # 1 768
        a2 = F.relu(self.act_fc3(a2)).unsqueeze(1) # 1 20

        a2_pre = F.relu(self.act_fea_cv3(img_pre))  # 1 32 19 10
        a2_pre = F.relu(self.act_fea_cv4(a2_pre))  # 1 64 8 4
        a2_pre = F.relu(self.act_fea_cv5(a2_pre))  # 1 64 6 2
        a2_pre = a2_pre.reshape(a2_pre.shape[0], -1)  # 1 768
        a2_pre = F.relu(self.act_fc3(a2_pre)).unsqueeze(1)  # 1 20

        a2 = a2 + self.alpha * a2_pre

        # other
        a3 = torch.cat((goal, speed, orientation), dim=-1)
        a3 = F.relu(self.act_fc4(a3)).unsqueeze(1) # 1 20

        # concat
        a4 = torch.cat((a, a2, a3), dim=1) # 1 80
        a4 = self.sa_encoder(a4)
        a4 = a4.flatten(start_dim=1)
        a4 = F.relu(self.act_fc2(a4))
        mean1 = F.sigmoid(self.actor1(a4))
        mean2 = F.tanh(self.actor2(a4))
        mean = torch.cat((mean1, mean2), dim=-1)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        # laser
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v)).unsqueeze(1)

        # camera
        v2 = F.relu(self.crt_fea_cv3(img))
        v2 = F.relu(self.crt_fea_cv4(v2))
        v2 = F.relu(self.act_fea_cv5(v2))
        v2 = v2.reshape(v2.shape[0], -1)
        v2 = F.relu(self.crt_fc3(v2)).unsqueeze(1)

        v2_pre = F.relu(self.crt_fea_cv3(img_pre))
        v2_pre = F.relu(self.crt_fea_cv4(v2_pre))
        v2_pre = F.relu(self.act_fea_cv5(v2_pre))
        v2_pre = v2_pre.reshape(v2_pre.shape[0], -1)
        v2_pre = F.relu(self.crt_fc3(v2_pre)).unsqueeze(1)

        v2 = v2 + self.alpha * v2_pre

        # other
        v3 = torch.cat((goal, speed, orientation), dim=-1)
        v3 = F.relu(self.crt_fc4(v3)).unsqueeze(1)

        # concat
        v4 = torch.cat((v, v2, v3), dim=1)
        v4 = self.sa_encoder(v4)
        v4 = v4.flatten(start_dim=1)
        v4 = F.relu(self.crt_fc2(v4))
        v4 = self.critic(v4)

        return v4, action, logprob, mean

    def evaluate_actions(self, x, img, img_pre, goal, speed, orientation, action):
        v, _, _, mean = self.forward(x, img, img_pre, goal, speed, orientation)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


if __name__ == '__main__':
    print("test")
    # from torch.autograd import Variable

    # net = MLPPolicy(3, 2)

    # observation = Variable(torch.randn(2, 3))
    # v, action, logprob, mean = net.forward(observation)
    # print(v)