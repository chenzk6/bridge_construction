import numpy as np  
import torch  
from utils.make_env_utils import make_env, get_env_kwargs  
from vec_env.subproc_vec_env import SubprocVecEnv  
from utils.wrapper import VecPyTorch  
from torch_algorithms import PPO_dev  
from torch_algorithms.policies import MultiDiscreteAttentionPolicy  
from utils.evaluation import evaluate_fixed_scene  

def custom_visualization():  


    
    # 使用与训练命令完全相同的参数  
    env_kwargs = get_env_kwargs(  
        env_id="FetchBridgeBullet7Blocks-v1",  
        horizon=50,  
        random_size=True,  # 禁用随机尺寸  
        min_num_blocks=7,  
        action_scale=0.6,  
        restart_rate=0.0,  
        noop=True,  
        robot="lh",  
        force_scale=10,  
        adaptive_primitive=False  
    )  
    
    # 处理max_episode_steps  
    max_episode_steps = env_kwargs.get("max_episode_steps", None)  
    env_kwargs.pop("max_episode_steps", None)  
    
    env_kwargs.update({  
        "need_visual": True,  
        "render": True,  
        "primitive": True,  
        "compute_path": True,  
    })  
    
    # 创建环境  
    def make_thunk(rank):  
        return lambda: make_env("FetchBridgeBullet7Blocks-v1", rank, 0, max_episode_steps, None,  
                                done_when_success=False, reward_scale=1.0, bonus_weight=0.0,  
                                env_kwargs=env_kwargs)  
    
    eval_env = SubprocVecEnv([make_thunk(0)], reset_when_done=False)  
    eval_env = VecPyTorch(eval_env, torch.device("cpu"))  
    
    # 复制run.py中的策略创建逻辑  
    aux_head = False  # 评估时不需要辅助头  
    
    # 使用与训练时相同的参数（从你的训练命令推断）  
    args_num_bin = [64, 64, 64]  # 默认值，可能需要调整  
    args_hidden_size = 64  # 从错误信息推断  
    args_n_attention_blocks = 3  # 从错误信息推断  
    args_n_heads = 1  # 默认值  
    args_v_ensemble = 1  # 默认值  
    args_bilevel_action = False  # 默认值  
    
    policy = MultiDiscreteAttentionPolicy(  
        eval_env.observation_space.shape,   
        7,  # num_blocks  
        eval_env.action_space.shape[0] - 1,   
        num_bin=args_num_bin,  
        feature_dim=args_hidden_size,   
        n_attention_blocks=args_n_attention_blocks,  
        object_dim=eval_env.get_attr("object_dim")[0],  
        has_cliff=eval_env.get_attr("has_cliff")[0],   
        aux_head=aux_head,  
        arch="shared",   
        base_kwargs={'n_heads': args_n_heads},  
        noop=True,   
        n_values=args_v_ensemble,  
        bilevel_action=args_bilevel_action  
    )  
    policy.to(torch.device("cpu"))  
    policy.eval()  
    
    # 创建模型  
    model = PPO_dev(eval_env, policy, torch.device("cpu"), auxiliary_task="inverse_dynamics")  
    model.load("logs/lh_train_1_after/final.pt", eval=True)  
    
    # 设置固定场景参数  
    # initial_positions = [[0.5, -0.2, 0.1] for _ in range(7)]  

    # obs0 = eval_env.env_method("get_obs")[0]
    # initial_positions = [obs0[17 * i: 17 * i + 3].tolist() for i in range(7)]

    #积木位置、姿态
    num_blocks = 7
    initial_positions = [
        eval_env.env_method("get_block_reset_pos", i)[0].tolist()
        for i in range(num_blocks)
    ]

    yaws = [0.0, 0.0, 0.0, 1.57, 1.57, 1.57, 0.78]
    initial_orientations = [
    [0.0, 0.0, np.sin(y / 2.0), np.cos(y / 2.0)]
    for y in yaws
    ]
    # initial_positions = [
    #     [0.75, 0.10, 0.025],
    #     [0.85, 0.10, 0.025],
    #     [0.95, 0.10, 0.025],
    #     [0.75, 1.1, 0.025],
    #     [0.85, 1.1, 0.025],
    #     [0.95, 1.1, 0.025],
    #     [1.0, 0.6, 0.025],
    # ]
    initial_positions = [
        [0.75, 0.10, 0.025],
        [0.85, 0.10, 0.025],
        [0.95, 0.10, 0.025],
        [0.75, 1.1, 0.025],
        [0.85, 0.3, 0.025],
        [0.95, 0.8, 0.025],
        [1.0, 0.6, 0.025],
    ]

    # 手动输入积木长度  
    # manual_block_lengths = [0.1, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13]  
    manual_block_lengths = [0.1, 0.1, 0.1, 0.0868, 0.0534, 0.1185, 0.1245]
    # manual_block_lengths = [0.1, 0.1, 0.1, 0.1, 0.05, 0.12, 0.05]
    object_sizes = [[0.025, length, 0.025] for length in manual_block_lengths]  
    # cliff0_center = 0.278
    # cliff1_center = 0.979
    cliff0_center = 0.328
    cliff1_center = 1.029
    
    # 运行一次episode  
    evaluate_fixed_scene(  
        eval_env, initial_positions, object_sizes,   
        cliff0_center, cliff1_center, model.policy, torch.device("cpu"),
        initial_orientations=initial_orientations
    )  

if __name__ == "__main__":  
    custom_visualization()