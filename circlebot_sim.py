from circlebot_env import CircleBotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import imageio.v2 as iio
import torch.nn as nn
"""
This program trains and tests the circlebot env with parameters for which stage of the curriculum as well as 
whether or not to use the existing weights, and whether to train or just evaluate
"""

#avoids using lamda when creating subprocenv which is not pickleable
def make_env(stage_n, render):
    def _init():
        return CircleBotEnv(stage = stage_n, render= render)
    return _init

#Configure what stage to run
    #stage 1: learn to go towards target with obstacles present at the sides and avoid obstacles at low frequency
    #stage 2: learn to avoisd obstacles at high frequency
    #stage 3: learn to deal with starting angle changes
 
stage_num = 1
#Configure if you want to repeat training on top of current weights
repeat = True
#Configure if you want to train or just re-evaluate
train = False


if __name__  == "__main__":    
    
    if train:
        train_env = SubprocVecEnv([make_env(stage_n = stage_num, render = False) for _ in range(16)])
        if stage_num == 1 and not repeat:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                device="cpu",
                n_steps=1024,
                batch_size=512,
                learning_rate=3e-4,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs = dict(
                    net_arch=[512, 256, 256, 128],
                    activation_fn=nn.ReLU
                )
            )
        elif not repeat:
            train_env = VecNormalize.load("circlebot_vecnorm"+str(stage_num-1)+".pkl", train_env)
            model = PPO.load("ppo_stage"+str(stage_num-1), env = train_env, device="cpu")
        else:
            train_env = VecNormalize.load("circlebot_vecnorm"+str(stage_num)+".pkl", train_env)
            model = PPO.load("ppo_stage"+str(stage_num), env = train_env, device="cpu")


        model.learn(total_timesteps=200000)
        model.save("ppo_stage"+str(stage_num))
        train_env.save("circlebot_vecnorm"+str(stage_num)+".pkl")
        train_env.close()
        
        


    # Evaluate result as mp4 video

    eval_env = DummyVecEnv([lambda: CircleBotEnv(stage = stage_num, render=True)])
    eval_env = VecNormalize.load("circlebot_vecnorm"+str(stage_num)+".pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load("ppo_stage"+str(stage_num), env=eval_env, device="cpu")

    obs = eval_env.reset()  
    raw_env = eval_env.envs[0]  

    frames = []
    for _ in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)  
        frame = raw_env.render(mode="rgb_array")            
        frames.append(frame)
        if dones[0]:
            break

    # Save MP4
    with iio.get_writer("robot_sim"+str(stage_num)+".mp4", fps=30, codec="libx264") as writer:
        for f in frames:
            writer.append_data(f)
