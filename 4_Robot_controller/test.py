import os
import argparse
from datetime import datetime
from agent import Agent
from env.env import PandaEnv

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_type', type=str, default='CQL-SacVae')
	parser.add_argument('--memory_size', type=int, default=1e5)
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--gui', action='store_true')
	parser.add_argument('--init_latent', action='store_false')
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--num_rollout', type=int, default=10)
	parser.add_argument("--gpu", default='0', type=str)

	args = parser.parse_args()

	# Configs which are constant across all tasks.
	configs = {
		'env_type': args.env_type,
		'num_steps': 3000000,
		'init_latent' : args.init_latent,
		'initial_latent_steps': 40000,
		'batch_size': 256,
		'latent_batch_size': 16,
		'beta': 1e-6,
		'policy_lr': 3e-5,
		'critic_lr': 3e-4,
		'latent_lr': 1e-4,
		'latent_dim': 256,
		'hidden_units': [512, 512, 512, 512, 512],#[1024, 1024, 1024],
		'memory_size': args.memory_size,
		'gamma': 0.99,
		'target_update_interval': 1,
		'tau': 5e-3,
		'entropy_tuning': True,
		'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
		'leaky_slope': 0.2,
		'grad_clip': None,
		'updates_per_step': 1,
		'learning_log_interval': 100,
		'eval_interval': 1000,
		'cuda': args.cuda,
		'seed': args.seed,
		'gpu' : args.gpu,
		'num_rollout' : args.num_rollout
	}

	env = PandaEnv(gui=args.gui)

	log_dir = os.path.join(
		'logs', args.env_type,
		f'cql-seed{args.seed}--{datetime.now().strftime("%Y%m%d-%H%M")}')

	agent = Agent(env=env, log_dir=log_dir, **configs)
	agent.test()

if __name__ == '__main__':
	run()
