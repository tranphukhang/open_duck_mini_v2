from pathlib import Path

import mujoco
import numpy as np

from mujoco_kinematics import get_site_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

XML_PATH = PROJECT_ROOT / "xmls" / "scene_flat_terrian_v3.xml"

print("XML path:", XML_PATH)


# ---------------------------------------------------------
# Load MuJoCo model
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)


q_right = np.deg2rad([
    1.0,       # hip yaw
    2.0,        # hip roll
    3.0,      # hip pitch
    4.0,       # knee
    5.0,      # ankle
])


T = get_site_transform(
    model=model,
    data=data,
    leg_name="right",
    q_leg=q_right,
    target_site="R6",
    reference_site="R0",
)


np.set_printoptions(
    precision=6,
    suppress=True
)

print("T_R1_R6 =")
print(T)