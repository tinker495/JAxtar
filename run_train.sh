tmux new -s train -d "python train_davi.py --puzzle rubikscube --shuffle_length 30 --reset -l 0.2"
tmux new -s train -d "python train_qlearning.py --puzzle rubikscube --shuffle_length 30 --reset -l 0.2"
#tmux new -s train -d "python train_davi.py --puzzle lightsout --shuffle_length 30 --reset"
#tmux new -s train -d "python train_davi.py --puzzle n-puzzle --shuffle_length 500"
