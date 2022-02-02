def vec_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    rollout_function_kwargs=None
):
    num_envs = env.n_envs

    policy_obs = env.reset()
    agent.reset(policy_obs)

    observations = [policy_obs]
    rewards = [np.zeros(num_envs)]
    actions = [np.zeros((num_envs, env_actions_space.low.size))]
    terminals = [[False] * num_envs]
    agent_info = [{}]
    env_info = [{}]

    for step in range(0, max_path_length):
        action, agent_info = agent.get_action(policy_obs)

        obs, reward, done, info = env.step()