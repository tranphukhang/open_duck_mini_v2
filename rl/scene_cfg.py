from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import GeomCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from rl.robot_cfg import FOOT_GEOM_NAMES
from mjlab.scene import SceneCfg
from rl.robot_cfg import get_robot_cfg


def get_terrain_cfg() -> TerrainEntityCfg:
    """Return the flat terrain configuration."""

    return TerrainEntityCfg(
        terrain_type="plane",
        geoms=(GeomCfg(geom_names_expr=(r"^terrain$",), contype=1, conaffinity=1, condim=3, priority=1, friction=(0.8, 0.005, 0.0001),),),
    )

def get_contact_switch_cfg() -> ContactSensorCfg:
    """Return the binary foot contact sensor configuration."""

    return ContactSensorCfg(
        name="feet_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot",),
        secondary=ContactMatch(mode="body", pattern="terrain",),
        fields=("found",),
        reduce="mindist",
        num_slots=1,
        track_air_time=True,
    )

def get_contact_wrench_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_wrench",
        primary=ContactMatch(
            mode="geom",
            pattern=FOOT_GEOM_NAMES,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force", "torque", "pos"),
        reduce="netforce",
        num_slots=1,
    )

def get_scene_cfg(num_envs: int = 1, env_spacing: float = 2.0,) -> SceneCfg:
    """Return the Open Duck Mini V2 scene configuration."""

    return SceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        terrain=get_terrain_cfg(),
        entities={"robot": get_robot_cfg(),},
        sensors=(get_contact_switch_cfg(), get_contact_wrench_cfg(),),
    )