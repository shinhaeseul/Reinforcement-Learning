import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

#Hyperparameters
learning_rate = 0.001
gamma         = 0.92

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 4)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # https://sanghyu.tistory.com/113 (learning rate scheduler 소개)
        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
        #                                 lr_lambda=lambda epoch: 0.95 ** epoch,
        #                                 last_epoch=-1,
        #                                 verbose=False) # ~ 0.4
        # self.scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=self.optimizer,
        #                                         lr_lambda=lambda epoch: 0.95 ** epoch) # ~ 0.25
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5) # ~ 0
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []
    
    def learning_rate_scheduler(self):
        self.scheduler.step()
        print("lr: ", self.optimizer.param_groups[0]['lr'])

def main():
    # env = gym.make('FrozenLake-v1', render_mode="human")
    env = gym.make('FrozenLake-v1', is_slippery=False)
    pi = Policy()
    score = 0.0
    scores=[]
    print_interval = 1000
    epochs=100
    for epoch in range(epochs):
        for n_epi in range(1000): # 10000 episode
            s, _ = env.reset() # 처음 상태의 observation
            done = False
            while not done: # FrozenLake-v1 forced to terminates at 500 step.
                prob = pi(torch.tensor(s, dtype=torch.float).unsqueeze(0))
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, truncated, info = env.step(a.item())
                pi.put_data((r,prob[a]))
                s = s_prime
                score += r
                # env.render()
            pi.train_net()

            if (epoch*1000+n_epi)%print_interval==0 and (epoch*1000+n_epi)!=0:
                scores.append(score/print_interval)
                print("# of episode :{}, avg score : {}".format((epoch*1000+n_epi), score/print_interval))
                score = 0.0
        pi.learning_rate_scheduler()
    env.close()
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Score per Episode')
    plt.show()
if __name__ == '__main__':
    main()
