echo $1
nohup python -u train.py --env $1 > logs/td3_$1.log 2>&1 &