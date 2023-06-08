# Q Actor-Critic
# gradient J = E[gradient pi(s,a)*Q^w(s,a)]

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
alpha = 0.0002
beta = 0.0002
gamma = 0.98
n_rollout = 10

class QActorCritic(nn.Module):
    def __init__(self):
        super(QActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_q = nn.Linear(256,1)
        self.optimizer_theta = optim.Adam(self.parameters(), lr=alpha)
        self.weights = nn.Parameter(torch.zeros((2, 4)))  # 가중치 초기화 action_num x feature_num
        self.optimizer_weights = optim.Adam([self.weights], lr=beta)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def q(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc_q(x)
        return q
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst,a_prime_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,a_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            a_prime_lst.append(a_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch,a_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(a_prime_lst, dtype=torch.float),torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, a_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, a_prime, done = self.make_batch()
        td_target = r + gamma * self.q(s_prime, a_prime) * done
        delta = td_target - self.q(s, a)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.q(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         

def main():  
    env = gym.make('CartPole-v1')
    model = QActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s, _ = env.reset()
        
        prob = model.pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample().item()
        
        while not done:
            for t in range(n_rollout):
                s_prime, r, done, truncated, info = env.step(a)
                
                prob_prime = model.pi(torch.from_numpy(s_prime).float())
                m_prime = Categorical(prob_prime)
                a_prime = m_prime.sample().item()
                
                model.put_data((s,a,r,s_prime,a_prime,done))
                
                s = s_prime
                a = a_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()