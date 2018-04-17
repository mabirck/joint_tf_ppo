# joint_pytorch-a2c-ppo-acktr
Extension of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr, making it feasible to run train on multiple games simultaneously.


#### Joint PPO

```bash
python main.py --env-name "SonicTheHedgehog-Genesis-SpringYardZone.Act3"  --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1
```
