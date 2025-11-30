# Robust-RL

## Environment variables configuration:
```Bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/username/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia"
```

## Tools

Converting tb log to npy sotrage:

> Robust_RL/PDSAC/utils/tbevent_to_npy.py

Draw converted npy:

> Robust_RL/PDSAC/utils/draw_npy.py

Example:
```Bash
LOGDIR="./PDSAC/train_result"
python tbevent_to_npy.py LOGDIR
python draw_from_npy.py LOGDIR
```
