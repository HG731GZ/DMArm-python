"""读取同构臂的 URDF 材质与外观网格，将零件顶点转换到各自的 CAD 连杆系。"""

from xml.etree import ElementTree

import numpy as np

from Robot_UR import CAD_LINKS


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
        self.links = [links[name] for name in CAD_LINKS]

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
