import copy
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

create_rollout_function = partial


def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
):
    if full_o_postprocess_func:

        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)

    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths["observations"] = paths["observations"][observation_key]
    return paths


def contextual_rollout(
    env,
    agent,
    observation_key=None,
    context_keys_for_policy=None,
    obs_processor=None,
    **kwargs
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ["context"]

    if not obs_processor:

        def obs_processor(o):
            combined_obs = [o[observation_key]]
            for k in context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)

    paths = rollout(env, agent, preprocess_obs_for_policy_fn=obs_processor, **kwargs)
    return paths


def rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )


@torch.no_grad()
def vec_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    terminals = np.array(terminals)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    env_infos = [
        [
            {key: env_infos[j][key][i] for key in env_infos[j]}
            for j in range(max_path_length)
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)
    paths = []
    for i in range(env.num_envs):
        paths.append(
            dict(
                observations=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                terminals=terminals[i],
                agent_infos=agent_infos,
                env_infos=env_infos[i],
            )
        )
    return paths


@torch.no_grad()
def rollout_modular(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x

    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    planner_raw_obs = []
    planner_raw_next_obs = []
    planner_observations = []
    planner_actions = []
    planner_rewards = []
    planner_terminals = []
    planner_agent_infos = []
    planner_env_infos = []
    planner_next_observations = []

    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    current_low_level_rewards = []
    while path_length < max_path_length:
        use_planner = agent.current_policy_str == "policy1"
        if use_planner and len(observations) > 0:
            # planner next obs should come after the low level policy finishes executing
            planner_next_observations.append(next_o)
            planner_raw_next_obs.append(next_o)
            planner_rewards[-1] += current_low_level_rewards.sum()
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        path_length += 1

        if use_planner:
            current_low_level_rewards = []
            planner_raw_obs.append(o)
            planner_observations.append(o)
            planner_rewards.append(r)
            planner_terminals.append(d)
            planner_actions.append(a)
            planner_agent_infos.append(agent_info)
            planner_env_infos.append(env_info)

            # low level policy next obs should come after planner policy executes
            # only add to next obs if there is a low level policy execution already, planner usually comes first
            if len(observations) > 0:
                next_observations[-1] = next_o
                raw_next_obs[-1] = next_o
                rewards[-1] += r  # add planner reward to low level policy reward
        else:
            raw_obs.append(o)
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            current_low_level_rewards.append(r)
            next_observations.append(next_o)
            raw_next_obs.append(next_o)

        o = next_o
    # planner next obs should come after the low level policy finishes executing
    # TODO: do this more cleanly, currently this code assumes we end on a low level policy execution
    planner_next_observations.append(next_o)
    planner_raw_next_obs.append(next_o)
    planner_rewards[-1] += sum(current_low_level_rewards)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)

    planner_actions = np.array(planner_actions)
    if len(planner_actions.shape) == 1:
        planner_actions = np.expand_dims(planner_actions, 1)
    planner_observations = np.array(planner_observations)
    planner_next_observations = np.array(planner_next_observations)
    if return_dict_obs:
        planner_observations = planner_raw_obs
        planner_next_observations = planner_raw_next_obs
    planner_rewards = np.array(planner_rewards)
    if len(planner_rewards.shape) == 1:
        planner_rewards = planner_rewards.reshape(-1, 1)
    terminals = np.array(terminals)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    env_infos = [
        [
            {key: env_infos[j][key][i] for key in env_infos[j]}
            for j in range(len(env_infos))
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    planner_terminals = np.array(planner_terminals)
    planner_observations = [planner_observations[:, i] for i in range(env.num_envs)]
    planner_next_observations = [
        planner_next_observations[:, i] for i in range(env.num_envs)
    ]
    planner_actions = [planner_actions[:, i] for i in range(env.num_envs)]
    planner_rewards = [planner_rewards[:, i] for i in range(env.num_envs)]
    planner_terminals = [planner_terminals[:, i] for i in range(env.num_envs)]
    planner_env_infos = [
        [
            {key: planner_env_infos[j][key][i] for key in planner_env_infos[j]}
            for j in range(len(planner_env_infos))
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)
    paths = []
    for i in range(env.num_envs):
        terminals[i][-1] = True
        planner_terminals[i][-1] = True
        paths.append(
            dict(
                observations=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                terminals=terminals[i],
                agent_infos=agent_infos,
                env_infos=env_infos[i],
                planner_observations=planner_observations[i],
                planner_next_observations=planner_next_observations[i],
                planner_actions=planner_actions[i],
                planner_rewards=planner_rewards[i],
                planner_terminals=planner_terminals[i],
                planner_agent_infos=planner_agent_infos,
                planner_env_infos=planner_env_infos[i],
            )
        )
    return paths


def deprecated_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
