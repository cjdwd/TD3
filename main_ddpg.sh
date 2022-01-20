echo $1
nohup python -u train_ddpg.py --env $1 > logs/ddpg_$1.log 2>&1 &