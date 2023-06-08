#REINFORCE 
import gym
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.gamma = 0.99

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train(self):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + R*self.gamma
            loss = -log_prob * R
            self.optimizer.zero_grad()
            loss.backward() # gradient 계산
            self.optimizer.step() # gradient를 통해 업데이트
        self.data = []


def main():
    env = gym.make('CartPole-v0')
    pi = Policy()
    avg_t = 0

    for n_epi in range(1000): # 10000 episode
        obs = env.reset() # 처음 상태의 observation(=state(4차원 벡터))
        for t in range(600): # 한 에피소드 내에서 600 iteration (참고:500스텝까지 가면 성공)
            obs = torch.tensor(obs, dtype=torch.float) # numpy array->torch의 tensor로 변환(네트워크 input으로 넣기 위함)
            out = pi(obs) # policy(state)=확률분포(action은 좌/우로 힘을 줄 두 확률(stochastic policy))
            m = Categorical(out) # torch에서 지원하는 Categorical(=확률분포) model
            action = m.sample() # 확률에 비례한 action(tensor(0) or tensor(1))이 나옴
            
            # 다음 state, reward, 마지막 state인지에 대한 정보, 다른 detail
            obs, r, done, info = env.step(action.item()) # environment에 action(tensor(scala action)->scala action)을 줌
            
            pi.put_data((r,torch.log(out[action]))) # policy 안에 reward,log(pi_theta(s,a)) pair data를 모음
            if done:
                break
        avg_t += t
        pi.train()
        if n_epi%20==0 and n_epi!=0:
            print("# of episode :{}, Avg timestep : {}".format(n_epi, avg_t/20.0))
            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()