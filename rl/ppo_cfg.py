from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def get_ppo_cfg() -> RslRlOnPolicyRunnerCfg:

    return RslRlOnPolicyRunnerCfg(
        seed=42,

        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),

        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),

        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),

        num_steps_per_env=24,
        max_iterations=30_000,
        save_interval=50,

        obs_groups={
            "actor": ("actor",),
            "critic": ("critic",),
        },

        experiment_name="open_duck_mini_v2",
        run_name="flat_locomotion",

        logger="tensorboard",
        upload_model=False,
    )