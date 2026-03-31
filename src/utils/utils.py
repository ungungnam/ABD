def map_vlm_action_to_vector(vlm_action):
    LANG_TO_VEC = {
        "front": [1, 0, 0],
        "back": [-1, 0, 0],
        "left": [0, -1, 0],
        "right": [0, 1, 0],
        "up": [0, 0, 1],
        "down": [0, 0, -1]
    }
    return LANG_TO_VEC[vlm_action]