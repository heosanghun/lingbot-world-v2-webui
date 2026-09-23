import math
import numpy as np
from scipy.spatial.transform import Rotation


def generate_action_trajectory(
    action: str,
    frame_num: int = 17,
    step_size: float = 0.5,
    turn_angle_deg: float = 12.0,
    width: int = 832,
    height: int = 480,
    fx: float = 502.9,
    fy: float = 503.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate camera poses and intrinsics for a given user movement action.
    
    Camera Coordinate System: OpenCV
    - +X: Right
    - +Y: Down
    - +Z: Forward (pointing into the scene)
    
    Args:
        action: One of 'W', 'S', 'A', 'D', 'Q', 'E', 'SPACE', 'C', 'IDLE'
                or lowercase equivalents ('forward', 'backward', etc.)
        frame_num: Number of frames in the chunk (default: 17 for 1-second rollout)
        step_size: Translation magnitude in camera units
        turn_angle_deg: Rotation angle in degrees for turning actions
        width: Frame width
        height: Frame height
        fx, fy: Focal lengths
        
    Returns:
        poses: [frame_num, 4, 4] float32 array of camera-to-world (c2w) matrices
        intrinsics: [frame_num, 4] float32 array of [fx, fy, cx, cy]
    """
    action = action.upper().strip()
    cx = width / 2.0
    cy = height / 2.0

    # Intrinsics constant across frames
    intrinsics = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (frame_num, 1))

    # Smooth ease-in-out factor: 0.5 * (1 - cos(pi * t))
    t = np.linspace(0.0, 1.0, frame_num)
    smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))

    # Base translation and rotation angles (in radians)
    dx = np.zeros(frame_num, dtype=np.float32)
    dy = np.zeros(frame_num, dtype=np.float32)
    dz = np.zeros(frame_num, dtype=np.float32)
    yaw = np.zeros(frame_num, dtype=np.float32)    # rotation around Y axis
    pitch = np.zeros(frame_num, dtype=np.float32)  # rotation around X axis

    rad_angle = np.deg2rad(turn_angle_deg)

    if action in ('W', 'FORWARD', 'UP_ARROW'):
        # Forward: move along +Z
        dz = smooth_t * step_size
    elif action in ('S', 'BACKWARD', 'DOWN_ARROW'):
        # Backward: move along -Z
        dz = -smooth_t * step_size
    elif action in ('A', 'LEFT', 'LEFT_ARROW'):
        # Strafe Left: move along -X with subtle yaw turn to the left
        dx = -smooth_t * step_size
        yaw = -smooth_t * (rad_angle * 0.4)
    elif action in ('D', 'RIGHT', 'RIGHT_ARROW'):
        # Strafe Right: move along +X with subtle yaw turn to the right
        dx = smooth_t * step_size
        yaw = smooth_t * (rad_angle * 0.4)
    elif action in ('Q', 'TURN_LEFT', 'PAN_LEFT'):
        # Turn Left: pure yaw rotation to the left
        yaw = -smooth_t * rad_angle
    elif action in ('E', 'TURN_RIGHT', 'PAN_RIGHT'):
        # Turn Right: pure yaw rotation to the right
        yaw = smooth_t * rad_angle
    elif action in ('SPACE', 'ASCEND', 'UP'):
        # Ascend: in OpenCV +Y is down, so -Y is UP
        dy = -smooth_t * step_size
    elif action in ('C', 'DESCEND', 'DOWN'):
        # Descend: move along +Y
        dy = smooth_t * step_size
    elif action in ('IDLE', 'X', 'STAY'):
        # Subtle organic handheld drift
        dx = np.sin(2 * np.pi * t) * (step_size * 0.05)
        dy = np.cos(2 * np.pi * t) * (step_size * 0.03)
        dz = np.zeros_like(t)
    else:
        # Default forward
        dz = smooth_t * step_size

    poses = np.zeros((frame_num, 4, 4), dtype=np.float32)
    for i in range(frame_num):
        # Rotation: Euler Y-X (yaw then pitch)
        rot = Rotation.from_euler('yx', [yaw[i], pitch[i]], degrees=False).as_matrix()
        poses[i, :3, :3] = rot
        poses[i, :3, 3] = [dx[i], dy[i], dz[i]]
        poses[i, 3, 3] = 1.0

    return poses, intrinsics
