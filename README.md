## Get started
Prerequisites:
* Python==3.9.17

```bash
# core dependencies
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-atari.txt
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/DLR-RM/stable-baselines3
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
pip install -e .[cuda]
conda install -c nvidia cuda-python
cd ..
cd cleanrl
cd sam_track
bash script/install.sh
bash script/download_ckpt.sh
cd FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```


# Download necessary ckpts to cleanrl/sam_track/ckpt

[Track Model](https://drive.google.com/file/d/1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq/view)

[FastSAM](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view)

# Train INSIGHT

To generate dataset, use
```bash
cd ..
python demo.py --video-name PongNoFrameskip-v4
```
To train cnn
```bash
cd ..
python train_cnn.py --wandb-project-name nsrl-eql --env-id PongNoFrameskip-v4 --run-name benchmark-pretrain-Pong-seed1 --seed 1
```
Or you can use a build-in [dataset](https://drive.google.com/file/d/1E_b3eBJ47ze1OJ7Nz1khsJ-q1YrcjTdu/view?usp=sharing) directly

To train policy, use 
```bash
python train_policy_atari.py --wandb-project-name nsrl-eql --env-id PongNoFrameskip-v4 --run-name benchmark-ng-reg-weight-1e-3-Pong-seed1 --ng True --reg_weight 1e-3 --seed 1 --load_cnn True
```

To train metadrive, use 
```bash
python train_policy_metadrive.py --wandb-project-name nsrl-eql --run-name benchmark-INSIGHT-MetaDriveEnv-seed1 --env-id MetaDriveEnv --cnn_loss_weight 2 --distillation_loss_weight 1 --load_cnn True --seed 1 --learning-rate 5e-5 --clip-coef 0.2 --ent-coef 0.01 --ego_state True --num-envs 8 --num-steps 125 --update-epochs 4 --num-minibatches 10 --max-grad-norm 0.5 --anneal-lr False --kl-penalty-coef 0.2 --reg_weight 1e-4  --use_eql_actor True
```
