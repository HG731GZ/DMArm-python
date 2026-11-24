"""在同一世界坐标系内绘制实心主端、半透明从端和两端的 DH6 坐标轴。"""

from itertools import product
from math import cos, radians, sin

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, sip

from ArmMesh import ArmMesh


# Qt 的 OpenGL 函数封装使用标准枚举值。
GL_FLOAT, GL_TRIANGLES, GL_LINES = 0x1406, 0x0004, 0x0001
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND = 0x0B71, 0x0B44, 0x0BE2
GL_LESS, GL_EQUAL = 0x0201, 0x0202
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x0302, 0x0303
GL_POLYGON_OFFSET_FILL = 0x8037

VERTEX_SHADER = """
attribute vec3 position;
attribute vec3 normal;
attribute vec3 material;
uniform mat4 projection;
uniform mat4 viewFromLink;
uniform vec4 tint;
uniform bool shaded;
varying vec4 color;
void main() {
    color = vec4(material * tint.rgb, tint.a);
    if (shaded) {
        vec3 n = normalize((viewFromLink * vec4(normal, 0.0)).xyz);
        vec3 light = normalize(vec3(-0.45, 0.65, 0.72));
        float diffuse = max(dot(n, light), 0.0);
        float shine = pow(max(dot(n, normalize(light + vec3(0.0, 0.0, 1.0))), 0.0), 28.0);
        color.rgb = color.rgb * (0.48 + 0.52 * diffuse) + 0.14 * shine;
    }
    gl_Position = projection * viewFromLink * vec4(position, 1.0);
}
"""
FRAGMENT_SHADER = """
varying vec4 color;
void main() { gl_FragColor = color; }
"""

MASTER_TINT = (0.25, 0.60, 0.76)
SLAVE_TINT = (1.0, 0.55, 0.17)
PREVIEW_TINT = (0.83, 0.88, 0.91)
AXIS_COLORS = np.array(((179, 60, 50), (37, 118, 63), (52, 93, 178))) / 255
WORLD_AXES_M, END_AXES_M = 0.13, 0.085
GRID = np.array([segment for t in np.arange(-6, 7) / 10
                 for segment in (((t, -.6, 0), (t, .6, 0)), ((-.6, t, 0), (.6, t, 0)))])


def line_vertices(points, colors):
    vertices = np.zeros((len(points), 9), dtype=np.float32)
    vertices[:, :3] = points
    vertices[:, 6:] = colors
    return vertices


def axis_vertices(stops):
    points = np.eye(3)[:, None, :] * np.array(stops)[None, :, None]
    return line_vertices(points.reshape(-1, 3), np.repeat(AXIS_COLORS, len(stops), axis=0))


def qt_matrix(matrix):
    return QtGui.QMatrix4x4(matrix.ravel().tolist())


class ArmView(QtWidgets.QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        surface = QtGui.QSurfaceFormat()
        surface.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
        surface.setVersion(2, 1)
        surface.setDepthBufferSize(24)
        surface.setSamples(4)
        self.setFormat(surface)
        self.master = self.slave = None
        self.drag_pos = QtCore.QPoint()
        self.reset_camera()

    def set_model(self, urdf_path, slave_opacity):
        """主窗在首次显示前指定两端共用的模型和从端不透明度。"""
        self.arm = ArmMesh(urdf_path)
        self.slave_opacity = slave_opacity

    def initializeGL(self):
        profile = QtGui.QOpenGLVersionProfile()
        profile.setVersion(2, 0)
        self.gl = self.context().versionFunctions(profile)
        self.gl.initializeOpenGLFunctions()
        self.program = QtGui.QOpenGLShaderProgram()
        for kind, source in ((QtGui.QOpenGLShader.Vertex, VERTEX_SHADER),
                             (QtGui.QOpenGLShader.Fragment, FRAGMENT_SHADER)):
            if not self.program.addShaderFromSourceCode(kind, source):
                raise RuntimeError(self.program.log())
        for index, name in enumerate(("position", "normal", "material")):
            self.program.bindAttributeLocation(name, index)
        if not self.program.link():
            raise RuntimeError(self.program.log())
        self.projection_uniform = self.program.uniformLocation("projection")
        self.pose_uniform = self.program.uniformLocation("viewFromLink")
        self.tint_uniform = self.program.uniformLocation("tint")
        self.shaded_uniform = self.program.uniformLocation("shaded")

        # 所有零件只在建立图形上下文时读入。两台臂共用一个静态顶点缓冲，每节单独变换。
        blocks = list(self.arm.vertex_blocks())
        boxes = []
        for block in blocks:
            xyz = block[:, :3]
            boxes.append(list(product(*zip(xyz.min(axis=0), xyz.max(axis=0)))))
        self.boxes = np.ones((len(blocks), 8, 4))
        self.boxes[:, :, :3] = boxes
        blocks.extend((line_vertices(GRID.reshape(-1, 3), (.85, .89, .91)),
                       axis_vertices((0, 1)),
                       axis_vertices((0, .18, .27, .45, .55, .73, .82, 1))))
        first = 0
        ranges = []
        for block in blocks:
            ranges.append((first, len(block)))
            first += len(block)
        self.ranges = ranges[:-3]
        self.grid, self.solid_axes, self.dashed_axes = ranges[-3:]
        data = np.concatenate(blocks)
        self.buffer = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
        self.buffer.create()
        self.buffer.bind()
        self.buffer.allocate(sip.voidptr(data.ctypes.data), data.nbytes)
        self.buffer.release()
        self.context().aboutToBeDestroyed.connect(self.release_gl)

    def release_gl(self):
        self.makeCurrent()
        self.buffer.destroy()
        self.program = None
        self.doneCurrent()

    def camera_matrices(self, arms, axes):
        az, el = radians(self.azimuth), radians(self.elevation)
        right = np.array((-sin(az), cos(az), 0))
        back = np.array((cos(el) * cos(az), cos(el) * sin(az), sin(el)))
        view = np.eye(4)
        view[:3, :3] = (right, np.cross(back, right), back)

        # 用两端实体的包围盒与三轴端点共同取景，旋转后也保留完整的基座和示教器。
        points = [pose[:3, 3] + length * pose[:3, :3].T for pose, length, _ in axes]
        for poses, _, _ in arms:
            corners = self.boxes @ poses.transpose(0, 2, 1)
            points.append(corners[:, :, :3].reshape(-1, 3))
        xy = np.concatenate(points) @ view[:2, :3].T
        low, high = xy.min(axis=0), xy.max(axis=0)
        viewport = np.array((self.width(), self.height()))
        scale = self.zoom * np.min((viewport - 32) / np.maximum(high - low, .18))
        projection = np.eye(4)
        projection[0, 0], projection[1, 1] = 2 * scale / viewport
        projection[:2, 3] = -(low + high) * scale / viewport
        projection[2, 2] = -.5
        return view, projection

    def paintGL(self):
        arms = []
        axes = [(self.arm.assembly[0], WORLD_AXES_M, self.solid_axes)]
        if self.master is not None or self.slave is None:
            poses = self.arm.assembly if self.master is None else self.arm.link_poses(self.master)
            arms.append((poses, PREVIEW_TINT if self.master is None else MASTER_TINT, 1.0))
            axes.append((poses[-1], END_AXES_M, self.solid_axes))
        if self.slave is not None:
            poses = self.arm.link_poses(self.slave)
            arms.append((poses, SLAVE_TINT, self.slave_opacity))
            axes.insert(1, (poses[-1], END_AXES_M, self.dashed_axes))

        view, projection = self.camera_matrices(arms, axes)
        gl = self.gl
        gl.glClearColor(.953, .969, .976, 1)
        gl.glDepthMask(True)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glDisable(GL_BLEND)
        self.program.bind()
        self.buffer.bind()
        for index in range(3):
            self.program.enableAttributeArray(index)
            self.program.setAttributeBuffer(index, GL_FLOAT, index * 12, 3, 36)
        self.program.setUniformValue(self.projection_uniform, qt_matrix(projection))
        self.program.setUniformValue(self.shaded_uniform, 0)
        self.program.setUniformValue(self.tint_uniform, 1.0, 1.0, 1.0, 1.0)
        self.draw_lines(view, self.grid, 1.0)

        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_CULL_FACE)
        gl.glDepthFunc(GL_LESS)
        self.program.setUniformValue(self.shaded_uniform, 1)
        for poses, tint, alpha in arms:
            matrices = [qt_matrix(m) for m in view @ poses]
            self.program.setUniformValue(self.tint_uniform, *tint, alpha)
            if alpha < 1:
                # 先求从端可见外表面的深度，再混合一次颜色，避免内部零件叠成实心。
                # 微小深度偏移让完全重合的表面显示主端材质。
                gl.glEnable(GL_POLYGON_OFFSET_FILL)
                gl.glPolygonOffset(0, 2)
                gl.glColorMask(False, False, False, False)
                self.draw_links(matrices)
                gl.glColorMask(True, True, True, False)
                gl.glDepthFunc(GL_EQUAL)
                gl.glDepthMask(False)
                gl.glEnable(GL_BLEND)
                gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.draw_links(matrices)
        gl.glColorMask(True, True, True, True)
        gl.glDepthMask(True)
        gl.glDepthFunc(GL_LESS)
        for flag in (GL_POLYGON_OFFSET_FILL, GL_BLEND, GL_CULL_FACE, GL_DEPTH_TEST):
            gl.glDisable(flag)

        # 坐标轴与网格共用相机，叠在表面上使基座内部和腕部的原点始终可读。
        self.program.setUniformValue(self.shaded_uniform, 0)
        self.program.setUniformValue(self.tint_uniform, 1.0, 1.0, 1.0, 1.0)
        for pose, length, span in axes:
            axis_view = view @ pose
            axis_view[:3, :3] *= length
            self.draw_lines(axis_view, span, 2.0 * self.devicePixelRatioF())
        for index in range(3):
            self.program.disableAttributeArray(index)
        self.buffer.release()
        self.program.release()

    def draw_lines(self, matrix, span, width):
        self.program.setUniformValue(self.pose_uniform, qt_matrix(matrix))
        self.gl.glLineWidth(width)
        self.gl.glDrawArrays(GL_LINES, *span)

    def draw_links(self, matrices):
        for matrix, (first, count) in zip(matrices, self.ranges):
            self.program.setUniformValue(self.pose_uniform, matrix)
            self.gl.glDrawArrays(GL_TRIANGLES, first, count)

    def reset_camera(self):
        self.azimuth, self.elevation, self.zoom = -145.0, 22.0, 1.0
        self.update()

    def mousePressEvent(self, event):
        self.drag_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            delta = event.pos() - self.drag_pos
            self.azimuth -= .45 * delta.x()
            self.elevation = max(-85, min(85, self.elevation + .45 * delta.y()))
            self.drag_pos = event.pos()
            self.update()

    def wheelEvent(self, event):
        self.zoom = max(.5, min(4.0, self.zoom * 1.15 ** (event.angleDelta().y() / 120)))
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_camera()
