python main.py distance-train qlearning --logger wandb --optimizer prodigy
python main.py distance-train qlearning --logger wandb --optimizer prodigy -pre quality
python main.py distance-train qlearning --logger wandb -nc '{"start_dim": 1000, "hidden_dim": 600, "hidden_N": 10}'
python main.py distance-train qlearning -dd --logger wandb -pre quality --loss asymmetric_huber --loss-args '{"asymmetric_tau": 0.05}'
python main.py distance-train davi -pre diffusion_distance --optimizer prodigy -nc '{"use_swiglu": true, "activation": "silu"}' --logger wandb --loss logcosh
python main.py distance-train qlearning -pre diffusion_distance -nc '{"activation": "silu"}' --logger wandb --loss logcosh
python main.py distance-train davi -pre diffusion_distance --optimizer muon

python main.py distance-train davi -p rubikscube-uqtm -pre diffusion_distance --logger wandb --loss logcosh --optimizer adopt
python main.py distance-train qlearning -p rubikscube-uqtm -pre diffusion_distance --logger wandb --loss logcosh --optimizer adopt

python main.py distance-train qlearning -lr 1e-4 --logger wandb --optimizer ano -nc '{"norm_fn": "group"}'
python main.py distance-train qlearning -lr 1e-4 --logger wandb --optimizer ano -nc '{"norm_fn": "group"}' -tsw
python main.py distance-train qlearning -lr 1e-4 --logger wandb --optimizer ano -nc '{"norm_fn": "group", "resblock_fn": "preactivation"}'
python main.py distance-train qlearning -lr 1e-4 --logger wandb --optimizer ano -nc '{"norm_fn": "group", "resblock_fn": "preactivation"}' -tsw
python main.py distance-train qlearning -lr 1e-4 --logger wandb --optimizer ano -nc '{"norm_fn": "group", "resblock_fn": "preactivation", "Res_N": 6}'
python main.py distance-train qlearning -nc '{"use_swiglu": true, "activation": "silu", "norm_fn": "group", "resblock_fn": "preactivation", "use_shortcut": true}' --logger wandb
python main.py distance-train qlearning -nc '{"use_swiglu": true, "activation": "silu", "norm_fn": "group", "resblock_fn": "preactivation", "use_shortcut": true}' --logger wandb --optimizer adopt
python main.py distance-train qlearning -nc '{"resblock_fn": "preactivation"}' --logger wandb

git clone https://github.com/tinker495/JAxtar.git -b feature/new-nets && cd JAxtar && pip install -r requirements_gpu.txt

python main.py distance-train spr_qlearning --logger wandb
python main.py distance-train spr_qlearning -p rubikscube-random --logger wandb

python main.py distance-train spr_qlearning -pre quality --logger wandb
python main.py distance-train spr_qlearning -p rubikscube-random -pre quality --logger wandb
