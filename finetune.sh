#!/bin/bash

MODEL_PATH=./models/trained_designer.pt
LH_MODEL_PATH=./logs/lh_finetune
USE_LH_MODEL=true

if [ "$USE_LH_MODEL" = true ]; then
	# 查找目录下最新的.pt文件
	LATEST_PT=$(find "$LH_MODEL_PATH" -maxdepth 1 -name "*.pt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
	if [ -n "$LATEST_PT" ]; then
		MODEL_PATH=$LATEST_PT
		echo "Using latest .pt file found: $MODEL_PATH"
	else
		echo "Warning: No .pt file found in $LH_MODEL_PATH, using default path"
		MODEL_PATH=$LH_MODEL_PATH/lh_trained_designer.pt
	fi
fi

# 把log.txt内容追加到log_backup.txt
if [ "$USE_LH_MODEL" = true ]; then
	# 先检查下log_backup.txt是否存在，若不存在则创建
	if [ ! -f $LH_MODEL_PATH/log_backup.txt ]; then
		touch $LH_MODEL_PATH/log_backup.txt
	fi
	# 检查下log.txt是否存在，若存在则追加内容，否则跳过追加
	if [ ! -f $LH_MODEL_PATH/log.txt ]; then
		echo "Warning: log.txt not found in $LH_MODEL_PATH, skipping backup"
	else
		echo "Backing up log.txt to log_backup.txt"
		cat $LH_MODEL_PATH/log.txt >>$LH_MODEL_PATH/log_backup.txt
	fi

	# 对progress.csv做同样的备份操作
	if [ ! -f $LH_MODEL_PATH/progress_backup.csv ]; then
		touch $LH_MODEL_PATH/progress_backup.csv
	fi
	if [ ! -f $LH_MODEL_PATH/progress.csv ]; then
		echo "Warning: progress.csv not found in $LH_MODEL_PATH, skipping backup"
	else
		echo "Backing up progress.csv to progress_backup.csv"
		cat $LH_MODEL_PATH/progress.csv >>$LH_MODEL_PATH/progress_backup.csv
	fi
fi

python run.py --env_id FetchBridgeBullet7Blocks-v1 --random_size --action_scale 0.6 --noop \
	--algo ppg --policy_arch shared --num_workers 72 --num_timesteps 1e7 \
	--noptepochs 10 --aux_freq 1 --inf_horizon \
	--restart_rate 0.5 --priority_type td --priority_decay 0.0 --filter_priority 0.9 --clip_priority \
	--auxiliary_task inverse_dynamics --force_scale 10 --primitive --robot lh \
	--load_path $MODEL_PATH --log_path $LH_MODEL_PATH
