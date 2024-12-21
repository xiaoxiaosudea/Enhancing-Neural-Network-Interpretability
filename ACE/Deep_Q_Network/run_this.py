"""
run_this 是更新步骤
"""
from maze_env import Maze
from RL_brain import DeepQNetwork
import time
import optim_1
import optim

def run_maze():
    step = 0 #记录走到了第几步，因为刚开始还不能学习，先要在记忆库中存储一些记忆
    d = 0
    dic_his_action = []
    epoches = 500
    for episode in range(epoches):
        # 初始化observation
        observation = env.reset()
        his_action = []
        print(f"epoch is : {episode}")
        while True:
            # 更新环境 
            env.render()

            # 通过观测observation，随机选择一个动作
            action = RL.choose_action(observation)
            if len(his_action) == 10:
                print("未找到比最大组合更小的组合")
                d += 1
                break
            if action in his_action:
                continue
            his_action.append(action)
            # 将动作输入环境中，得到下一个observation、奖励、游戏是否结束
            observation_, reward, done = env.step(action)

            print(f"observation_:{observation_}-----reward:{reward}--------action:{action}------observation:{observation}")
            # with open("D:\\机器学习\\强化学习\\莫烦Python\\Reinforcement-learning-with-tensorflow-master\\Reinforcement-learning-with-tensorflow-master\\contents\\5_Deep_Q_Network\\run_this_.txt", mode='a', encoding="utf-8") as F:
            #     # F.write(f"epoch{episode}\nobservation:{observation}\naction:{action}\nobservation_:{observation_}\nreward:{reward}\nstep:{step}\n\n")
            #     F.write(f"epoch{episode}\nobservation_:{observation_}-----reward:{reward}--------action:{action}------observation:{observation}----------step:{step}\n\n")
            RL.store_transition(observation, action, reward, observation_)  #存储记忆，当前的环境状态、动作、奖励和下一个状态

            if (step > 200) and (step % 5 == 0): #200步之后再开始学习，200步之后每隔5步更新一次参数                
                RL.learn()

            observation = observation_  # 更新环境

            
            if done:    # 如果游戏结束则进入下一个回合
                break
            step += 1
        dic_his_action.append(his_action)


    
    print('game over')   # game over741
    env.destroy()
    if d == epoches:
        print("没有能够改变预测结果的组合")
    sorted_dic_his_action = sorted(dic_his_action, key=lambda x: len(x))
    for i in range(len(sorted_dic_his_action)):
        if i != 0:
            if len(sorted_dic_his_action[i]) != len(sorted_dic_his_action[i-1]):
                print(sorted_dic_his_action[i])
            else:
                break
        else:
            print(sorted_dic_his_action[i])

if __name__ == "__main__":
    # maze game
    # env = Maze()
    # env = optim.optim()
    env = optim_1.optim()
    # print(env.n_features)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                       output_graph=False
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()