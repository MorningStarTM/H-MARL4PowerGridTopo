iconfig = {"ENV_NAME" : "l2rpn_case14_sandbox",
            "middle_agent_type" : "capa",  # Options: "capa", "fixed_sub"
            "agent_type" : "ppo",  # Options: "ppo", "dqn", etc.
            "input_dim" : 128,
            'has_continuous_action_space': False,
            'action_std_init': 0.6,
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'gamma': 0.99,
            'K_epochs': 80,
            'eps_clip': 0.2,
            }