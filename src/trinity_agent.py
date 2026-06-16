
from trinity_critic import TrinityCritic
from trinity_encoder import TrinityEncoder


class TrinityAgent():
    def __init__(self, 
                 observation_dim = None, 
                 #action_dim = None,
                 critic_lr = 0.01,
                 critic_resolution = 40,
                 critic_conscience_factor = 0.5,
                 critic_conscience_lr = 0.01,
                 critic_prior_ema_alpha = 0.001,
                 critic_visualize = False,
                 critic_viz_update_interval = 100,
                 feature_names = None,
                 encoder_learning_rate = 0.01,
                 encoder_semantic_codelength = 8,
                 encoder_resolution = 10,
                 encoder_conscience_factor = 0.5,
                 encoder_conscience_lr = 0.01,
                 encoder_prior_ema_alpha = 0.001,
                 encoder_visualize = False,
                 encoder_viz_update_interval = 100,
                 ):
        
        self.critic = TrinityCritic(
            n_inputs=observation_dim,
            resolution=critic_resolution,
            lr=critic_lr,
            conscience_factor=critic_conscience_factor,
            conscience_lr=critic_conscience_lr,
            prior_ema_alpha=critic_prior_ema_alpha,
            critic_visualize=critic_visualize,
            critic_viz_update_interval=critic_viz_update_interval,
            feature_names=feature_names,
        )

        self.encoder = TrinityEncoder(
            observation_dim=observation_dim,
            learning_rate=encoder_learning_rate,
            semantic_codelength=encoder_semantic_codelength,
            encoder_resolution=encoder_resolution,
            conscience_factor=encoder_conscience_factor,
            conscience_lr=encoder_conscience_lr,
            prior_ema_alpha=encoder_prior_ema_alpha,
            encoder_visualize=encoder_visualize,
            encoder_viz_update_interval=encoder_viz_update_interval,
            feature_names=feature_names,
        )

        #self.local_actor = TrinityLocalActor(observation_dim, action_dim)

        # FUTURE WORK: Implement a TrinityRemoteActor to act as a remote actor with access to a world model
        # for higher-resolution predictions and planning. Initially the remote actor receives both raw
        # observations and semantic codes.  As the remote actor trains its encoder, raw observations
        # are throttled to zero unless an anomalous, out-of-distribution observation is detected,
        # in which case the remote actor can request raw observations until the anomaly passes. 
        # 
        # self.remote_actor  = TrinityRemoteActor(observation_dim, action_dim)
    
    def step(self, observation, instruction):
        # Step the critic to get the first semantic bit (z0) and other diagnostic outputs
        ensemble_prediction, ensemble_score, scores, pmi_values, per_predictions = self.critic.step(observation, instruction)

        # Step the encoder with the observation and z0 to get the full semantic code
        semantic_code = self.encoder.step(observation, ensemble_prediction)

        # The local actor can then use the semantic code for action selection (not implemented yet)
        # action = self.local_actor.select_action(semantic_code)

        return semantic_code, ensemble_prediction, ensemble_score, scores, pmi_values, per_predictions