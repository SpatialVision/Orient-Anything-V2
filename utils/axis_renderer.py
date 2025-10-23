import bpy
import math
import os
from utils.paths import *

class BlendRenderer:
    def __init__(self, blend_file_path=RENDER_FILE):
        """
        初始化渲染器，加载指定的 .blend 文件并进行基础设置。
        
        :param blend_file_path: 要加载的 .blend 文件的完整路径
        """
        if not os.path.isfile(blend_file_path):
            raise FileNotFoundError(f"Blend file not found: {blend_file_path}")

        # 加载 blend 文件
        bpy.ops.wm.open_mainfile(filepath=blend_file_path)

        # 设置渲染引擎为 Cycles
        bpy.context.scene.render.engine = 'CYCLES'

        # 使用 CPU 渲染
        bpy.context.scene.cycles.device = 'CPU'

        # 设置采样数为 4
        bpy.context.scene.cycles.samples = 4

        # 设置所有反弹次数为 4（包括 diffuse, glossy, transmission, etc.）
        bpy.context.scene.cycles.max_bounces = 4

        # 设置渲染分辨率
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.scene.render.resolution_percentage = 100

        # 启用透明背景（RGBA）
        bpy.context.scene.render.film_transparent = True

        # 遍历所有对象，初始化渲染可见性
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = False
            elif obj.type == 'CAMERA':
                obj.hide_render = False
            elif obj.type == 'MESH':
                obj.hide_render = True  # 默认所有网格不参与渲染

        # 设置活动摄像机（选第一个）
        cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
        if cameras:
            bpy.context.scene.camera = cameras[0]

        print(f"Loaded blend file: {blend_file_path}")
        print("Render settings applied: 512x512, CPU, samples=4, bounces=4, transparent background.")
        
        self.alpha_axis_map = {
            0: "单轴平面",
            1: "三轴",
            2: "双向标注",
            4: "四向标注"
        }
        

    def _get_all_children(self, obj):
        """递归获取对象的所有子对象（包括嵌套子级）"""
        children = []
        for child in obj.children:
            children.append(child)
            children.extend(self._get_all_children(child))
        return children

    def render_axis(self, azi, ele, rot, alpha, save_path):
        """
        渲染特定方向的图像。
        
        :param azi: 方位角（绕 Z 轴旋转，弧度）
        :param ele: 仰角（绕 Y 轴旋转，弧度）
        :param rot: 自转（绕 X 轴旋转，弧度）
        :param save_path: 渲染结果保存路径（如 '/output/render.png'）
        """
        # 遍历所有对象，初始化渲染可见性
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = False
            elif obj.type == 'CAMERA':
                obj.hide_render = False
            elif obj.type == 'MESH':
                obj.hide_render = True  # 默认所有网格不参与渲染
        # 根据 alpha 选择目标对象
        target_name = self.alpha_axis_map.get(alpha, "单轴平面")
        target_obj = None
        for obj in bpy.data.objects:
            # if obj.type == 'MESH' and obj.name == target_name:
            if obj.name == target_name:
                target_obj = obj
                break

        if target_obj is None:
            raise ValueError(f'Object named "{target_name}" not found in the scene.')

        # 获取该对象及其所有子对象
        all_objects_to_render = [target_obj] + self._get_all_children(target_obj)

        # 设置它们参与渲染
        for obj in all_objects_to_render:
            if obj.type == 'MESH':
                obj.hide_render = False

        # 设置旋转（ZYX 顺序：Z=azi, Y=ele, X=rot → Euler XYZ = (rot, ele, azi)）
        # 注意：Blender 使用弧度
        target_obj.rotation_mode = 'ZYX'  # 确保使用欧拉角 ZYX 模式
        target_obj.rotation_euler = (rot*math.pi/180, ele*math.pi/180, -azi*math.pi/180)

        # 确保路径目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 设置输出路径
        bpy.context.scene.render.filepath = save_path

        # 执行渲染并保存
        bpy.ops.render.render(write_still=True)

        print(f"Rendered and saved to: {save_path}")
        
        
if __name__ == "__main__":
    renderer = BlendRenderer(RENDER_FILE)
    # Example usage:
    renderer.render_axis(45, 0, 0, 1, "./test_demo_output/render_1_dir_azi45.png")
    renderer.render_axis(0, 45, 0, 2, "./test_demo_output/render_2_dir_ele45.png")
    renderer.render_axis(0, 0, 45, 4, "./test_demo_output/render_4_dir_rot45.png")
    # renderer.render_1_dir()
    # renderer.render_2_dir()
    # renderer.render_4_dir()


