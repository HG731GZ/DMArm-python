"""读取同构臂的 URDF 外观网格，将 CAD 连杆坐标系配准到控制侧 DH 坐标系。"""

from math import pi
from xml.etree import ElementTree

import numpy as np

from Robot_UR import Robot_UR


# CAD 装配时两根长连杆竖直，腕部指向世界 −Y。对应控制侧的 DH 角，单位 rad。
ASSEMBLY_DH_RAD = (0, -pi / 2, 0, -pi / 2, 0, 0)
STL_FACE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def urdf_origin(element):
    """将 URDF 的 xyz（m）、固定轴 rpy（rad）转换为局部到父系的齐次矩阵。"""
    origin = element.find("origin")
    rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    matrix = np.eye(4)
    matrix[:3, :3] = ((cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
                      (sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
                      (-sp, cp*sr, cp*cr))
    matrix[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
    return matrix


class ArmMesh:
    def __init__(self, urdf_path):
        self.directory = urdf_path.parent
        robot = ElementTree.parse(urdf_path).getroot()
        self.materials = {m.attrib["name"]: np.fromstring(m.find("color").attrib["rgba"], sep=" ")[:3]
                          for m in robot.findall("material")}
        links = {link.attrib["name"]: link for link in robot.findall("link")}
        rest = {"base_link": np.eye(4)}
        self.links = [links["base_link"]]
        for number in range(1, 7):
            joint = robot.find(f"joint[@name='joint{number}']")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            rest[child] = rest[parent] @ urdf_origin(joint)
            self.links.append(links[child])
        self.assembly = np.stack([rest[link.attrib["name"]] for link in self.links])

        # 控制基座位于第一关节中心。装配姿态下一次求出各 CAD 系在对应 DH 系中的固定变换。
        self.world_from_control = self.assembly[1]
        assembly_dh = self.world_from_control @ Robot_UR(ASSEMBLY_DH_RAD).transforms
        self.dh_from_link = np.linalg.solve(assembly_dh, self.assembly[1:])

    def link_poses(self, model):
        moving = self.world_from_control @ model.transforms @ self.dh_from_link
        return np.concatenate((self.assembly[:1], moving))

    def vertex_blocks(self):
        """逐连杆合并装配零件，输出位置、面法向与 CAD 材质颜色。"""
        meshes = {}
        for link in self.links:
            parts = []
            for visual in link.findall("visual"):
                path = self.directory / visual.find("geometry/mesh").attrib["filename"]
                if path not in meshes:
                    # 本模型的二进制 STL 头部为 80 字节说明和 4 字节面数。
                    meshes[path] = np.fromfile(path, dtype=STL_FACE, offset=84)
                faces = meshes[path]
                origin = urdf_origin(visual).astype(np.float32)
                rotation = origin[:3, :3].T
                vertices = np.empty((len(faces), 3, 9), dtype=np.float32)
                vertices[:, :, :3] = faces["vertices"] @ rotation + origin[:3, 3]
                vertices[:, :, 3:6] = (faces["normal"] @ rotation)[:, None, :]
                vertices[:, :, 6:] = self.materials[visual.find("material").attrib["name"]]
                parts.append(vertices.reshape(-1, 9))
            yield np.concatenate(parts)
