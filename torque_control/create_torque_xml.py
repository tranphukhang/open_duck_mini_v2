import os
import xml.etree.ElementTree as ET


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


INPUT_XML = os.path.join(
    ROOT,
    "xmls",
    "open_duck_mini_v2.xml"
)

OUTPUT_XML = os.path.join(
    ROOT,
    "xmls",
    "open_duck_mini_v2_torque.xml"
)


def convert_actuator_to_torque():

    print("="*70)
    print("CREATE TORQUE XML")
    print("="*70)

    print("Input:")
    print(INPUT_XML)

    tree = ET.parse(INPUT_XML)
    root = tree.getroot()


    actuator = root.find("actuator")

    if actuator is None:
        raise RuntimeError(
            "Cannot find actuator section"
        )


    old_actuators = list(actuator)

    print()
    print("Original actuators:")
    print(len(old_actuators))


    # remove position actuator
    for act in old_actuators:
        actuator.remove(act)


    # joint list
    torque_joints = [

        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",

        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",

        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",

    ]


    for name in torque_joints:

        motor = ET.SubElement(
            actuator,
            "motor"
        )

        motor.set(
            "name",
            name
        )

        motor.set(
            "joint",
            name
        )


    tree.write(
        OUTPUT_XML,
        encoding="utf-8",
        xml_declaration=True
    )


    print()
    print("Created:")
    print(OUTPUT_XML)

    print()
    print("Torque actuators:")
    for i,name in enumerate(torque_joints):
        print(i,name)


if __name__ == "__main__":

    convert_actuator_to_torque()