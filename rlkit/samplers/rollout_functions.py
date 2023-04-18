import copy
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
import traceback

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
    bad_masks = []
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
        bad_masks.append(env_info["bad_mask"])
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
    bad_masks = np.array(bad_masks)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    bad_masks = [bad_masks[:, i][:, 0, 0] for i in range(env.num_envs)]
    env_infos = [
        [
            {key: env_infos[j][key][i] for key in env_infos[j]}
            for j in range(max_path_length)
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    # convert all arrays to masked arrays
    for i in range(env.num_envs):
        mask = bad_masks[i]
        observations[i] = np.ma.array(
            observations[i],
            mask=mask.reshape(-1, 1).repeat(observations[i].shape[1], axis=1),
        )
        next_observations[i] = np.ma.array(
            next_observations[i],
            mask=mask.reshape(-1, 1).repeat(next_observations[i].shape[1], axis=1),
        )
        actions[i] = np.ma.array(
            actions[i], mask=mask.reshape(-1, 1).repeat(actions[i].shape[1], axis=1)
        )
        rewards[i] = np.ma.array(rewards[i], mask=mask)
        terminals[i] = np.ma.array(terminals[i], mask=mask)
        # NOTE: adding masks to env_infos does not work, because masking a scalar just makes it a null value
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
                bad_masks=bad_masks[i],
            )
        )
    return paths, paths


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
    from torchvision.utils import save_image

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
    bad_masks = []

    planner_raw_obs = []
    planner_raw_next_obs = []
    planner_observations = []
    planner_actions = []
    planner_rewards = []
    planner_terminals = []
    planner_agent_infos = []
    planner_env_infos = []
    planner_next_observations = []
    planner_bad_masks = []

    merged_raw_obs = []
    merged_raw_next_obs = []
    merged_observations = []
    merged_actions = []
    merged_rewards = []
    merged_terminals = []
    merged_agent_infos = []
    merged_env_infos = []
    merged_next_observations = []
    merged_bad_masks = []

    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    current_low_level_rewards = []
    episode_breaks = []
    terminate_each_stage = agent.terminate_each_stage
    while path_length < max_path_length:
        #     # planner next obs should come after the low level policy finishes executing
        #     planner_next_observations.append(next_o)
        #     planner_raw_next_obs.append(next_o)
        #     if len(current_low_level_rewards) > 0:
        #         planner_rewards[-1] += current_low_level_rewards.sum() / len(
        #             current_low_level_rewards
        #         )
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        # use_planner is updated after get_action call
        use_planner = agent.current_policy_str == "policy1"
        if use_planner and len(observations) > 0:
            episode_breaks.append(len(observations))
        # save_image(torch.tensor(np.array(env.env_method("get_image"))).permute(0, 3, 1, 2) / 255., f"{path_length}.png", nrow=4)
        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        path_length += 1

        if use_planner:
            if terminate_each_stage and len(observations) > 0:
                terminals[-1] = np.array([True] * env.num_envs)
            current_low_level_rewards = []
            planner_raw_obs.append(o)
            planner_observations.append(o)
            planner_rewards.append(r)
            planner_terminals.append(d)
            planner_actions.append(a)
            planner_agent_infos.append(agent_info)
            planner_env_infos.append(env_info)

            # TODO: testing back to the old setup
            planner_next_observations.append(next_o)
            planner_raw_next_obs.append(next_o)
            planner_bad_masks.append(env_info["bad_mask"])
            if terminate_each_stage:
                planner_terminals[-1] = np.array([True] * env.num_envs)

            # low level policy next obs should come after planner policy executes
            # only add to next obs if there is a low level policy execution already, planner usually comes first
            # if len(observations) > 0:
            #     next_observations[-1] = next_o
            #     raw_next_obs[-1] = next_o
            #     rewards[-1] += r  # add planner reward to low level policy reward
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
            bad_masks.append(env_info["bad_mask"])

        # merged:
        merged_raw_obs.append(o)
        merged_observations.append(o)
        merged_rewards.append(r)
        merged_terminals.append(d)
        merged_actions.append(a)
        merged_agent_infos.append(agent_info)
        merged_env_infos.append(env_info)
        merged_next_observations.append(next_o)
        merged_raw_next_obs.append(next_o)
        merged_bad_masks.append(env_info["bad_mask"])

        o = next_o
    if terminate_each_stage and len(observations) > 0:
        terminals[-1] = np.array([True] * env.num_envs)
    # planner next obs should come after the low level policy finishes executing
    # TODO: do this more cleanly, currently this code assumes we end on a low level policy execution
    # planner_next_observations.append(next_o)
    # planner_raw_next_obs.append(next_o)
    # if len(current_low_level_rewards) > 0:
    #     planner_rewards[-1] += sum(current_low_level_rewards) / len(
    #         current_low_level_rewards
    #     )  # want to re-scale rewards to be in the same range as the high level policy
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
    bad_masks = np.array(bad_masks)

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
    planner_bad_masks = np.array(planner_bad_masks)

    terminals = np.array(terminals)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    bad_masks = [bad_masks[:, i] for i in range(env.num_envs)]
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
    planner_bad_masks = [planner_bad_masks[:, i] for i in range(env.num_envs)]
    planner_env_infos = [
        [
            {key: planner_env_infos[j][key][i] for key in planner_env_infos[j]}
            for j in range(len(planner_env_infos))
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    # merged:
    merged_actions = np.array(merged_actions)
    if len(merged_actions.shape) == 1:
        merged_actions = np.expand_dims(merged_actions, 1)

    merged_raw_obs = np.array(merged_raw_obs)
    merged_observations = np.array(merged_observations)
    merged_next_observations = np.array(merged_next_observations)
    if return_dict_obs:
        merged_observations = merged_raw_obs
        merged_next_observations = merged_raw_next_obs
    merged_rewards = np.array(merged_rewards)
    if len(merged_rewards.shape) == 1:
        merged_rewards = merged_rewards.reshape(-1, 1)
    merged_bad_masks = np.array(merged_bad_masks)

    merged_terminals = np.array(merged_terminals)
    merged_observations = [merged_observations[:, i] for i in range(env.num_envs)]
    merged_next_observations = [
        merged_next_observations[:, i] for i in range(env.num_envs)
    ]
    merged_actions = [merged_actions[:, i] for i in range(env.num_envs)]
    merged_rewards = [merged_rewards[:, i] for i in range(env.num_envs)]
    merged_terminals = [merged_terminals[:, i] for i in range(env.num_envs)]
    merged_bad_masks = [merged_bad_masks[:, i] for i in range(env.num_envs)]
    merged_env_infos = [
        [
            {key: merged_env_infos[j][key][i] for key in merged_env_infos[j]}
            for j in range(len(merged_env_infos))
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    paths = []
    merged_paths = []
    only_keep_trajs_after_grasp_success = (
        agent.only_keep_trajs_after_grasp_success  # do NOT use episode breaks if we are only keeping trajs after grasp success
    )
    only_keep_trajs_stagewise = agent.only_keep_trajs_stagewise

    # convert all arrays to masked arrays
    for i in range(env.num_envs):
        mask = bad_masks[i]
        observations[i] = np.ma.array(
            observations[i],
            mask=mask.reshape(-1, 1).repeat(observations[i].shape[1], axis=1),
        )
        next_observations[i] = np.ma.array(
            next_observations[i],
            mask=mask.reshape(-1, 1).repeat(next_observations[i].shape[1], axis=1),
        )
        actions[i] = np.ma.array(
            actions[i], mask=mask.reshape(-1, 1).repeat(actions[i].shape[1], axis=1)
        )
        rewards[i] = np.ma.array(rewards[i], mask=mask)
        terminals[i] = np.ma.array(terminals[i], mask=mask)
        # NOTE: adding masks to env_infos does not work, because masking a scalar just makes it a null value

        mask = merged_bad_masks[i]
        merged_observations[i] = np.ma.array(
            merged_observations[i],
            mask=mask.reshape(-1, 1).repeat(merged_observations[i].shape[1], axis=1),
        )
        merged_next_observations[i] = np.ma.array(
            merged_next_observations[i],
            mask=mask.reshape(-1, 1).repeat(
                merged_next_observations[i].shape[1], axis=1
            ),
        )
        merged_actions[i] = np.ma.array(
            merged_actions[i],
            mask=mask.reshape(-1, 1).repeat(merged_actions[i].shape[1], axis=1),
        )
        merged_rewards[i] = np.ma.array(merged_rewards[i], mask=mask)
        merged_terminals[i] = np.ma.array(merged_terminals[i], mask=mask)

        mask = planner_bad_masks[i]
        planner_observations[i] = np.ma.array(
            planner_observations[i],
            mask=mask.reshape(-1, 1).repeat(planner_observations[i].shape[1], axis=1),
        )
        planner_next_observations[i] = np.ma.array(
            planner_next_observations[i],
            mask=mask.reshape(-1, 1).repeat(
                planner_next_observations[i].shape[1], axis=1
            ),
        )
        planner_actions[i] = np.ma.array(
            planner_actions[i],
            mask=mask.reshape(-1, 1).repeat(planner_actions[i].shape[1], axis=1),
        )
        planner_rewards[i] = np.ma.array(planner_rewards[i], mask=mask)
        planner_terminals[i] = np.ma.array(planner_terminals[i], mask=mask)

    for i in range(env.num_envs):
        merged_paths.append(
            dict(
                type="merged",
                observations=merged_observations[i],
                actions=merged_actions[i],
                rewards=merged_rewards[i],
                next_observations=merged_next_observations[i],
                terminals=merged_terminals[i],
                agent_infos=agent_infos,
                env_infos=merged_env_infos[i],
                bad_masks=merged_bad_masks[i],
            )
        )
        if agent.use_episode_breaks:
            prev_episode_break = 0
            for idx, episode_break in enumerate(episode_breaks):
                paths.append(
                    dict(
                        type="control",
                        observations=observations[i][prev_episode_break:episode_break],
                        actions=actions[i][prev_episode_break:episode_break],
                        rewards=rewards[i][prev_episode_break:episode_break],
                        next_observations=next_observations[i][
                            prev_episode_break:episode_break
                        ],
                        terminals=terminals[i][prev_episode_break:episode_break],
                        agent_infos=agent_infos,
                        env_infos=env_infos[i][prev_episode_break:episode_break],
                        bad_masks=bad_masks[i][prev_episode_break:episode_break],
                    )
                )
                paths.append(
                    dict(
                        type="planner",
                        observations=planner_observations[i][idx : idx + 1],
                        next_observations=planner_next_observations[i][idx : idx + 1],
                        actions=planner_actions[i][idx : idx + 1],
                        rewards=planner_rewards[i][idx : idx + 1],
                        terminals=planner_terminals[i][idx : idx + 1],
                        agent_infos=planner_agent_infos,
                        env_infos=planner_env_infos[i][idx : idx + 1],
                        bad_masks=planner_bad_masks[i][idx : idx + 1],
                    )
                )
                prev_episode_break = episode_break
            paths.append(
                dict(
                    type="control",
                    observations=observations[i][prev_episode_break:],
                    actions=actions[i][prev_episode_break:],
                    rewards=rewards[i][prev_episode_break:],
                    next_observations=next_observations[i][prev_episode_break:],
                    terminals=terminals[i][prev_episode_break:],
                    agent_infos=agent_infos,
                    env_infos=env_infos[i][prev_episode_break:],
                    bad_masks=bad_masks[i][prev_episode_break:],
                )
            )
            paths.append(
                dict(
                    type="planner",
                    observations=planner_observations[i][-1:],
                    next_observations=planner_next_observations[i][-1:],
                    actions=planner_actions[i][-1:],
                    rewards=planner_rewards[i][-1:],
                    terminals=planner_terminals[i][-1:],
                    agent_infos=planner_agent_infos,
                    env_infos=planner_env_infos[i][-1:],
                    bad_masks=planner_bad_masks[i][-1:],
                )
            )
        else:
            if only_keep_trajs_after_grasp_success and not env_infos[i][25]["grasped"]:
                paths.append(
                    dict(
                        type="control",
                        observations=observations[i][: episode_breaks[0]],
                        actions=actions[i][: episode_breaks[0]],
                        rewards=rewards[i][: episode_breaks[0]],
                        next_observations=next_observations[i][: episode_breaks[0]],
                        terminals=terminals[i][: episode_breaks[0]],
                        agent_infos=agent_infos,
                        env_infos=env_infos[i][: episode_breaks[0]],
                        bad_masks=bad_masks[i][: episode_breaks[0]],
                    )
                )
                paths.append(
                    dict(
                        type="planner",
                        observations=planner_observations[i][:1],
                        next_observations=planner_next_observations[i][:1],
                        actions=planner_actions[i][:1],
                        rewards=planner_rewards[i][:1],
                        terminals=planner_terminals[i][:1],
                        agent_infos=planner_agent_infos,
                        env_infos=planner_env_infos[i][:1],
                        bad_masks=planner_bad_masks[i][:1],
                    )
                )
            elif only_keep_trajs_stagewise:
                if planner_rewards[i][0] > 0.06:
                    if rewards[i][: episode_breaks[0]][-1] > 0.3:
                        if planner_rewards[i][1] > 0.6:
                            # add full traj
                            paths.append(
                                dict(
                                    type="control",
                                    observations=observations[i],
                                    actions=actions[i],
                                    rewards=rewards[i],
                                    next_observations=next_observations[i],
                                    terminals=terminals[i],
                                    agent_infos=agent_infos,
                                    env_infos=env_infos[i],
                                    bad_masks=bad_masks[i],
                                )
                            )
                            paths.append(
                                dict(
                                    type="planner",
                                    observations=planner_observations[i],
                                    next_observations=planner_next_observations[i],
                                    actions=planner_actions[i],
                                    rewards=planner_rewards[i],
                                    terminals=planner_terminals[i],
                                    agent_infos=planner_agent_infos,
                                    env_infos=planner_env_infos[i],
                                    bad_masks=planner_bad_masks[i],
                                )
                            )
                        else:
                            # only add second planner action, not second control traj
                            paths.append(
                                dict(
                                    type="control",
                                    observations=observations[i][: episode_breaks[0]],
                                    actions=actions[i][: episode_breaks[0]],
                                    rewards=rewards[i][: episode_breaks[0]],
                                    next_observations=next_observations[i][
                                        : episode_breaks[0]
                                    ],
                                    terminals=terminals[i][: episode_breaks[0]],
                                    agent_infos=agent_infos,
                                    env_infos=env_infos[i][: episode_breaks[0]],
                                    bad_masks=bad_masks[i][: episode_breaks[0]],
                                )
                            )
                            paths.append(
                                dict(
                                    type="planner",
                                    observations=planner_observations[i],
                                    next_observations=planner_next_observations[i],
                                    actions=planner_actions[i],
                                    rewards=planner_rewards[i],
                                    terminals=planner_terminals[i],
                                    agent_infos=planner_agent_infos,
                                    env_infos=planner_env_infos[i],
                                    bad_masks=planner_bad_masks[i],
                                )
                            )
                    else:
                        # add first planner action, first control traj
                        paths.append(
                            dict(
                                type="control",
                                observations=observations[i][: episode_breaks[0]],
                                actions=actions[i][: episode_breaks[0]],
                                rewards=rewards[i][: episode_breaks[0]],
                                next_observations=next_observations[i][
                                    : episode_breaks[0]
                                ],
                                terminals=terminals[i][: episode_breaks[0]],
                                agent_infos=agent_infos,
                                env_infos=env_infos[i][: episode_breaks[0]],
                                bad_masks=bad_masks[i][: episode_breaks[0]],
                            )
                        )
                        paths.append(
                            dict(
                                type="planner",
                                observations=planner_observations[i][:1],
                                next_observations=planner_next_observations[i][:1],
                                actions=planner_actions[i][:1],
                                rewards=planner_rewards[i][:1],
                                terminals=planner_terminals[i][:1],
                                agent_infos=planner_agent_infos,
                                env_infos=planner_env_infos[i][:1],
                                bad_masks=planner_bad_masks[i][:1],
                            )
                        )
                else:
                    # add first planner action only
                    paths.append(
                        dict(
                            type="planner",
                            observations=planner_observations[i][:1],
                            next_observations=planner_next_observations[i][:1],
                            actions=planner_actions[i][:1],
                            rewards=planner_rewards[i][:1],
                            terminals=planner_terminals[i][:1],
                            agent_infos=planner_agent_infos,
                            env_infos=planner_env_infos[i][:1],
                            bad_masks=planner_bad_masks[i][:1],
                        )
                    )
            else:
                paths.append(
                    dict(
                        type="control",
                        observations=observations[i],
                        actions=actions[i],
                        rewards=rewards[i],
                        next_observations=next_observations[i],
                        terminals=terminals[i],
                        agent_infos=agent_infos,
                        env_infos=env_infos[i],
                        bad_masks=bad_masks[i],
                    )
                )
                paths.append(
                    dict(
                        type="planner",
                        observations=planner_observations[i],
                        next_observations=planner_next_observations[i],
                        actions=planner_actions[i],
                        rewards=planner_rewards[i],
                        terminals=planner_terminals[i],
                        agent_infos=planner_agent_infos,
                        env_infos=planner_env_infos[i],
                        bad_masks=planner_bad_masks[i],
                    )
                )
    return paths, merged_paths


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
