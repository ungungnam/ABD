from task.task_family import TaskDefinition, ReversibleTaskPair


def build_task_pair_from_config(task_cfg) -> ReversibleTaskPair:
    """Build a ReversibleTaskPair from a Hydra task config."""
    obj = task_cfg.object
    loc_a = task_cfg.location_a
    loc_b = task_cfg.location_b

    forward = TaskDefinition(
        name=f"{obj}_to_{loc_b}",
        language_task=task_cfg.forward_task,
        task_type="pick_place",
        canonical_state={"object": obj, "location": loc_a, "target": loc_b},
    )
    reverse = TaskDefinition(
        name=f"{obj}_to_{loc_a}",
        language_task=task_cfg.reverse_task,
        task_type="pick_place",
        canonical_state={"object": obj, "location": loc_b, "target": loc_a},
    )
    return ReversibleTaskPair(forward=forward, reverse=reverse)
