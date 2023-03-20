from rlkit.policies.base import Policy


class StepBasedSwitchingPolicy(Policy):
    """
    A policy that switches between two underlying policies based on the number of steps taken.
    """

    def __init__(
        self, policy1, policy2, policy2_steps_per_policy1_step, use_episode_breaks=False
    ):
        """
        Initializes a new instance of the StepBasedSwitchingPolicy class.

        Args:
            policy1 (Policy): The first underlying policy.
            policy2 (Policy): The second underlying policy.
            policy2_path_length (int): The number of steps to take before switching to policy1.
        """
        self.policy1 = policy1
        self.policy2 = policy2
        self.policy2_steps_per_policy1_step = policy2_steps_per_policy1_step
        self.num_steps = 0
        self.current_policy = policy1
        self.current_policy_str = "policy1"
        self.current_policy2_steps = 0
        self.take_policy1_step = True
        self.use_episode_breaks = use_episode_breaks

    def get_action(self, observation):
        """
        Gets an action from the currently active underlying policy.

        Args:
            observation: An observation of the environment.

        Returns:
            An action to take in the environment.
        """
        if self.take_policy1_step:
            self.current_policy = self.policy1
            self.current_policy_str = "policy1"
            self.take_policy1_step = False
        else:
            self.current_policy = self.policy2
            self.current_policy_str = "policy2"
            self.current_policy2_steps += 1
            if self.current_policy2_steps == self.policy2_steps_per_policy1_step:
                self.current_policy2_steps = 0
                self.take_policy1_step = True
        action = self.current_policy.get_action(observation)
        self.num_steps += 1
        return action

    def reset(self):
        """
        Resets the underlying policies and sets the number of steps and current policy to their initial values.
        """
        self.policy1.reset()
        self.policy2.reset()
        self.num_steps = 0
        self.current_policy = self.policy1
        self.current_policy_str = "policy1"
