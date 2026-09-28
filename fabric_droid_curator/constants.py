from __future__ import annotations

EVENT_ORDER = (
    "motion_start",
    "contact_start",
    "stable_grasp",
    "lift_start",
    "detach_complete",
    "release_start",
    "release_complete",
    "retreat_complete",
)

ACTION_BRANCHES = (
    "remove",
    "leave",
    "legacy_green_left",
    "legacy_white_right",
    "unknown",
)

SEGMENT_TYPES = (
    "full_remove",
    "full_leave",
    "grasp_probe",
    "remove_to_basket",
    "leave_on_rack",
    "recovery",
)

INSTRUCTION_TEMPLATES = {
    "grasp_probe": {
        "id": "grasp_probe_v2",
        "prompt": "Reach for the fabric, grasp it, and hold it for inspection.",
    },
    "remove_to_basket": {
        "id": "remove_to_basket_v2",
        "prompt": "Remove the fabric from the rack and drop it into the basket.",
    },
    "leave_on_rack": {
        "id": "leave_on_rack_v1",
        "prompt": "Leave the fabric on the rack, release it, and return to the ready position.",
    },
    "full_remove": {
        "id": "full_remove_v1",
        "prompt": "Reach for the fabric, remove it from the rack, and drop it into the basket.",
    },
    "full_leave": {
        "id": "full_leave_v1",
        "prompt": "Reach for the fabric, leave it on the rack, and return to the ready position.",
    },
    "recovery": {
        "id": "recovery_v1",
        "prompt": "Recover the robot to a safe ready position.",
    },
}

VIDEO_STREAM_PATHS = {
    "external": "recordings/MP4/exterior_image_1_left.mp4",
    "wrist": "recordings/MP4/wrist_image_left.mp4",
    "gelsight": "tactile/gelsight_left.mp4",
}

TIMESTAMP_STREAM_PATHS = {
    "external": "recordings/timestamps/exterior_image_1_left.npz",
    "wrist": "recordings/timestamps/wrist_image_left.npz",
    "gelsight": "tactile/gelsight_left_timestamps.npy",
}
