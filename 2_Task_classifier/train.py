import os
import argparse
from datetime import datetime
from agent import Agent
from env.env import PandaEnv

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_type', type=str, default='SacDisctere')
	parser.add_argument('--memory_size', type=int, default=1e5)
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--gui', action='store_true')
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--num_rollout', type=int, default=21)
	parser.add_argument("--gpu", default=0, type=int)

	args = parser.parse_args()

	# Configs which are constant across all tasks.
	configs = {
		'env_type': args.env_type,
		'num_steps': 3000000,
		'batch_size': 256,
		'policy_lr': 3e-5,
		'critic_lr': 3e-4,
		'latent_lr': 1e-4,
		'latent_dim': 256,
		'hidden_units': [256, 256],
		'memory_size': args.memory_size,
		'gamma': 0.8,
		'multi_step': 1,
		'target_entropy_ratio': 0.98,
		'target_update_interval': 1,
		'use_per': False,
		'dueling_net': True,
		'log_interval': 100,
		'num_eval_steps': 100,
		'cuda': args.cuda,
		'seed': args.seed,
		'gpu' : args.gpu,
		'num_rollout' : args.num_rollout
	}

	# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu}" 

	env = PandaEnv(gui=False)

	log_dir = os.path.join(
		'logs', args.env_type,
		f'sacd-seed{args.seed}--{datetime.now().strftime("%Y%m%d-%H%M")}')

	agent = Agent(env=env, log_dir=log_dir, **configs)
	agent.run()


if __name__ == '__main__':
	run()
