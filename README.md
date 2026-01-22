# Bridge Construction

Code for paper: Learning Design and Construction with Varying-Sized Materials via Prioritized Memory Resets.
![7obj_seq](7obj_seq_sim.png)

## Installation

```
conda env create -f environment.yml
conda activate bridge
```

## Play with a trained model

Visualize a bridge designer trained with teleportation (objects are directly teleported to instructed poses, no robot motion involved):

```
CUDA_VISIBLE_DEVICES=-1 python run.py \
--env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
--algo ppg --policy_arch shared --inf_horizon --auxiliary_task inverse_dynamics --robot xarm \
--load_path models/trained_designer.pt --play
```

Visualize a model fine-tuned with a low-level motion generator:

```
CUDA_VISIBLE_DEVICES=-1 python run.py \
--env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
--algo ppg --policy_arch shared --inf_horizon --auxiliary_task inverse_dynamics --robot xarm \
--load_path models/finetuned.pt --play --primitive
```

## Train a model by yourself

Train a bridge design policy with prioritized memory reset (PMR) and auxiliary prediction task:

```
python run.py --env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
              --algo ppg --policy_arch shared --num_workers 64 --num_timesteps 2e7 \
              --noptepochs 10  --aux_freq 1 --inf_horizon \
              --restart_rate 0.5 --priority_type td --priority_decay 0.0 --filter_priority 0.9 --clip_priority \
              --auxiliary_task inverse_dynamics --force_scale 0 --robot xarm \
              --log_path <your/log/dir>
```

Fine-tune a pre-trained designer with a low-level motion generator:

```
python run.py --env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
              --algo ppg --policy_arch shared --num_workers 64 --num_timesteps 1e7 \
              --noptepochs 10 --aux_freq 1 --inf_horizon \
              --restart_rate 0.5 --priority_type td --priority_decay 0.0 --filter_priority 0.9 --clip_priority \
              --auxiliary_task inverse_dynamics --force_scale 10 --primitive --robot xarm \
              --load_path <path/to/your/pretrained/model>
```

训练灵猴机器人，在xarm的基础上微调

```bash
python run.py --env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
            --algo ppg --policy_arch shared --num_workers 64 --num_timesteps 1e7 \
            --noptepochs 10 --aux_freq 1 --inf_horizon \
            --restart_rate 0.5 --priority_type td --priority_decay 0.0 --filter_priority 0.9 --clip_priority \
            --auxiliary_task inverse_dynamics --force_scale 10 --primitive --robot lh \
            --load_path ./models/trained_designer.pt --log_path logs/lh_finetune
```

使用xarm的模型测试lh

```
CUDA_VISIBLE_DEVICES=-1 python run.py \
--env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
--algo ppg --policy_arch shared --inf_horizon --auxiliary_task inverse_dynamics --robot lh \
--load_path models/finetuned.pt --play --primitive
```

[IKFast 导出流程](https://fishros.org.cn/forum/topic/680/moveit-ikfast%E8%BF%90%E5%8A%A8%E5%AD%A6%E6%8F%92%E4%BB%B6%E9%85%8D%E7%BD%AE-%E6%9C%80%E8%AF%A6%E7%BB%86-%E6%B2%A1%E6%9C%89%E4%B9%8B%E4%B8%80)
[urdf转xml脚本](https://gist.github.com/ompugao/cdd2e3aa788fd9cbc97bc1b9275b9709)

```bash
# urdf转dae
rosrun collada_urdf urdf_to_collada LingHouUrdf3.urdf LH.dae

# dae精确小数点
rosrun moveit_kinematics round_collada_numbers.py LH.dae LH.dae 5
xml精确小数点
python round_xml_precision.py LR4_R560.robot.xml LR4_R560_4.robot.xml 4

# urdf转xml
python urdf/urdf_to_ravexml.py --urdf ./urdf/LingHouUrdf3.urdf --robotxmldir ./openrave_models/

# openrave显示xml模型
openrave LR4_R560_4.robot.xml

# 打印xml的关节信息
python2 /usr/local/bin/openrave-robot.py LR4_R560.robot.xml --info links	//xml文件
python2 /usr/local/bin/openrave-robot.py LH.dae --info links	//dae文件
python2 /usr/local/bin/openrave-robot.py LH.dae --info joints

# xml转ik求解器
python2 $(openrave-config --python-dir)/openravepy/_openravepy_/ikfast.py --robot=LR4_R560_4.robot.xml --iktype=transform6d --baselink=0 --eelink=7 --savefile=ikfast61.cpp --maxcasedepth 1

# dae转ik求解器
python2 `openrave-config --python-dir`/openravepy/_openravepy_/ikfast.py --robot=LH.dae --iktype=transform6d --baselink=0 --eelink=7 --savefile=$(pwd)/ikfastec41.cpp --maxcasedepth 1

# 编译命令
cd env/ikfastpy
python setup.py build_ext --inplace
```
