import copy
import numpy as np
import pdb

def vec_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    rollout_function_kwargs=None
):
    num_envs = env.n_envs

    policy_obs = env.reset()
    agent.reset()

    observations = [policy_obs]
    rewards = [np.zeros(num_envs)]
    actions = [np.zeros((num_envs, env.action_space.low.size))]
    terminals = [[False] * num_envs]
    agent_infos = [{}]
    env_infos = [{}]

    for step in range(0, max_path_length):
        action, agent_info = agent.get_action(policy_obs)
        pdb.set_trace()
        obs, reward, done, info = env.step(copy.deepcopy(action))
        
        observations.append(obs)
        rewards.append(reward)
        terminals.append(done)
        actions.append(action)
        agent_infos.append(agent_info)
        env_infos.append(info)

        if done.all():
            break
        
        policy_obs = obs

    actions = np.array(actions)

    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    
    env_info_final = {}

    for info in env_infos[1:]:
        for key, value in info.items():
            if key not in env_info_final:
                env_info_final[key] = []
            
            env_info_final[key].append(value)
    
    for key, value in env_info_final.items():
        env_info_final[key] = np.concatenate(value, 1)
    env_infos = env_info_final

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=np.array(terminals),
        agent_infos=agent_infos,
        env_infos=env_infos
    )