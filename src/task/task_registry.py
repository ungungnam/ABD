from task.task_family import TaskDefinition, ReversibleTaskPair, TaskSequence


def build_stack_cups_task_from_config(task_cfg) -> TaskSequence:
    """Build a TaskSequence for stack_cups from config.

    2-cup mode (cup_a + cup_b only):
        forward:  stack cup_a on cup_b
        reverse:  unstack cup_a back to original position

    3-cup mode (cup_a + cup_b + cup_c):
        forward:  stack cup_a on cup_b  →  stack cup_c on cup_a
        reverse:  unstack cup_c         →  unstack cup_a

    Expected config fields:
        cup_a, cup_b, tag_id_a, tag_id_b  : first two cups (required)
        cup_c, tag_id_c                   : third cup (optional)
    """
    cup_a = task_cfg.cup_a
    cup_b = task_cfg.cup_b
    tag_a = int(task_cfg.tag_id_a)
    tag_b = int(task_cfg.tag_id_b)

    has_third = hasattr(task_cfg, "cup_c") and hasattr(task_cfg, "tag_id_c")

    # Sentinel used for unstack tasks to trigger cache-based place logic
    _UNSTACK = [0, 0]

    # --- forward step 1: stack cup_a on cup_b ---
    stack_a_on_b = TaskDefinition(
        name=f"stack_{cup_a}_on_{cup_b}",
        language_task=f"pick up the {cup_a} cup and stack it on the {cup_b} cup",
        task_type="stack_cups",
        canonical_state={"pick": cup_a, "place": cup_b},
        pick_tag_id=tag_a,
        place_tag_id=tag_b,
        stack_step="forward_1",
    )

    if not has_third:
        # --- 2-cup: 1 forward + 1 reverse ---
        unstack_a = TaskDefinition(
            name=f"unstack_{cup_a}_from_{cup_b}",
            language_task=f"pick up the {cup_a} cup from the {cup_b} cup and place it in its original position",
            task_type="stack_cups",
            canonical_state={"pick": cup_a, "place": cup_b},
            pick_tag_id=tag_a,
            place_xy_offset=_UNSTACK,
            stack_step="reverse_1",
        )
        return TaskSequence(tasks=[stack_a_on_b, unstack_a], n_forward=1)

    # --- 3-cup mode ---
    cup_c = task_cfg.cup_c
    tag_c = int(task_cfg.tag_id_c)

    # forward step 2: stack cup_c on cup_a (cup_a is now on cup_b) — TERMINAL forward step
    stack_c_on_a = TaskDefinition(
        name="stack_cups_forward",
        language_task=f"pick up the {cup_c} cup and stack it on the {cup_a} cup",
        task_type="stack_cups",
        canonical_state={"pick": cup_c, "place": cup_a},
        pick_tag_id=tag_c,
        place_tag_id=tag_a,
        stack_step="forward_2",
        validation_question="Are all three cups stacked into one single stack, with none separate?",
    )

    # reverse step 1: unstack cup_c back to its original position
    unstack_c = TaskDefinition(
        name=f"unstack_{cup_c}",
        language_task=f"pick up the {cup_c} cup from the {cup_a} cup and place it in its original position",
        task_type="stack_cups",
        canonical_state={"pick": cup_c, "place": cup_a},
        pick_tag_id=tag_c,
        place_tag_id=tag_c,
        place_xy_offset=_UNSTACK,
        stack_step="reverse_1",
    )

    # reverse step 2: unstack cup_a back to its original position — TERMINAL reverse step
    unstack_a = TaskDefinition(
        name="stack_cups_reverse",
        language_task=f"pick up the {cup_a} cup from the {cup_b} cup and place it in its original position",
        task_type="stack_cups",
        canonical_state={"pick": cup_a, "place": cup_b},
        pick_tag_id=tag_a,
        place_tag_id=tag_a,
        place_xy_offset=_UNSTACK,
        stack_step="reverse_2",
        validation_question="Is any cup physically resting on top of another cup, rather than standing separately on the table?",
        # If purple tag (tag_a) is NOT visible → cup_c still on cup_a → reverse_1 failed
        # intermediate_check_tag_id=tag_a,
        # intermediate_check_skip_if_visible=False,
    )

    return TaskSequence(
        tasks=[stack_a_on_b, stack_c_on_a, unstack_c, unstack_a],
        n_forward=2,
    )


def build_open_drawer_task_from_config(task_cfg) -> ReversibleTaskPair:
    """Build a ReversibleTaskPair for the open_drawer task.

    forward: open  the drawer (pull handle -X by pull_dist)
    reverse: close the drawer (push handle +X by pull_dist)

    Expected config fields:
        tag_id   : AprilTag ID on the drawer (e.g. 7)
    """
    tag_id = int(task_cfg.tag_id)

    forward = TaskDefinition(
        name="open_drawer",
        language_task="grab the drawer handle and pull to open the drawer",
        task_type="open_drawer",
        canonical_state={"drawer": "open"},
        pick_tag_id=tag_id,
        stack_step="forward",
    )
    reverse = TaskDefinition(
        name="close_drawer",
        language_task="grab the drawer handle and push to close the drawer",
        task_type="open_drawer",
        canonical_state={"drawer": "closed"},
        pick_tag_id=tag_id,
        stack_step="reverse",
    )
    return ReversibleTaskPair(forward=forward, reverse=reverse)


def build_task_pair_from_config(task_cfg) -> ReversibleTaskPair:
    """Build a ReversibleTaskPair from a Hydra task config."""
    obj = task_cfg.object
    loc_a = task_cfg.location_a
    loc_b = task_cfg.location_b

    fwd_offset = list(task_cfg.forward_place_offset) if hasattr(task_cfg, "forward_place_offset") else None
    rev_offset = list(task_cfg.reverse_place_offset) if hasattr(task_cfg, "reverse_place_offset") else None

    forward = TaskDefinition(
        name=f"{obj}_to_{loc_b}",
        language_task=task_cfg.forward_task,
        task_type="pick_place",
        canonical_state={"object": obj, "location": loc_a, "target": loc_b},
        place_offset=fwd_offset,
    )
    reverse = TaskDefinition(
        name=f"{obj}_to_{loc_a}",
        language_task=task_cfg.reverse_task,
        task_type="pick_place",
        canonical_state={"object": obj, "location": loc_b, "target": loc_a},
        place_offset=rev_offset,
    )
    return ReversibleTaskPair(forward=forward, reverse=reverse)
