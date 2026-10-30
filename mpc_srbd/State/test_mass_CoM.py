from pathlib import Path

import mujoco

from mass_CoM import get_mass_CoM


# ============================================================
# Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

XML_PATH = (
    PROJECT_ROOT
    / "xmls"
    / "scene_flat_terrian_v3.xml"
)

if not XML_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy MuJoCo XML:\n{XML_PATH}"
    )


print("Loading MuJoCo model:")
print(XML_PATH)


# ============================================================
# Load model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)


# ============================================================
# Compute whole-body mass
# ============================================================

mass_CoM = get_mass_CoM(
    model=model,
    root_body_name="base_link"
)


# ============================================================
# Display
# ============================================================

print("\n========================================")
print("Whole-body mass")
print("========================================")

print(
    f"mass_CoM = {mass_CoM:.6f} kg"
)