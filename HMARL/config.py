iconfig = {"ENV_NAME" : "l2rpn_case14_sandbox",
            "middle_agent_type" : "capa",  # Options: "capa", "fixed_sub"
            "agent_type" : "ppo",  # Options: "ppo", "dqn", etc.
            "input_dim" : 467,
            'has_continuous_action_space': False,
            'action_std_init': 0.6,
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'gamma': 0.99,
            'K_epochs': 80,
            'eps_clip': 0.2,

            'max_ep_len': 1000,                       # Max timesteps per episode
            'max_training_timesteps': int(3e6),       # Total training steps before stopping
            'action_std_init':0.6,

            'print_freq': 1000 * 10,                  # Print avg reward every n timesteps
            'log_freq': 1000 * 2,                     # Log reward every n timesteps
            'save_model_freq': int(1e5), 

            'action_std': 0.6,                        # Initial std for action distribution
            'action_std_decay_rate': 0.05,            # Decay rate for action std
            'min_action_std': 0.1,                    # Minimum std after decay
            'action_std_decay_freq': int(2.5e5), 
            
            'model_path': "HMARL\\models"
            }