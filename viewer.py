#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""

import os

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader
import csv

from core import FixedNode, Shader, Mesh, Node, KeyFrames, KeyFrameControlNode, Viewer, Texture,\
    TexturedMesh, PhongMesh, Skybox
from transform import translate, rotate, scale, vec, quaternion_from_euler, quaternion_slerp


# -------------- Cylinder ----------------------------------------------------
class WoodenCylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self, shader, light_dir, light_pos):
        super().__init__()
        # just load cylinder from file
        self.add(*load_textured("castle/cylinder.obj", shader, "castle/wood.jpg", light_dir, light_pos))


# -------------- Catapulte --------------------------------------------------
class Catapult(Node):
    """Hierarchical catapulte"""
    def __init__(self, shader, light_dir, light_pos):

        x_axis = (1, 0, 0)
        y_axis = (0, 1, 0)
        z_axis = (0, 0, 1)
        super().__init__()
        base_front = Node(transform=scale(1, 12, 1))
        base_front.add(WoodenCylinder(shader, light_dir, light_pos))
        base_left = Node(transform=translate(-15, 10, 0) @ rotate(z_axis, 90) @ scale(1, 15, 1))
        base_left.add(WoodenCylinder(shader, light_dir, light_pos))
        base_right = Node(transform=translate(-15, -10, 0) @ rotate(z_axis, 90) @ scale(1, 15, 1))
        base_right.add(WoodenCylinder(shader, light_dir, light_pos))
        base_back = Node(transform=translate(-30, 0, 0) @ scale(1, 12, 1))
        base_back.add(WoodenCylinder(shader, light_dir, light_pos))
        left_front_wheel = Node(transform=translate(0, 12, 0) @ scale(5, 1, 5))
        left_front_wheel.add(WoodenCylinder(shader, light_dir, light_pos))
        right_front_wheel = Node(transform=translate(0, -12, 0) @ scale(5, 1, 5))
        right_front_wheel.add(WoodenCylinder(shader, light_dir, light_pos))
        left_back_wheel = Node(transform=translate(-30, 12, 0) @ scale(5, 1, 5))
        left_back_wheel.add(WoodenCylinder(shader, light_dir, light_pos))
        right_back_wheel = Node(transform=translate(-30, -12, 0) @ scale(5, 1, 5))
        right_back_wheel.add(WoodenCylinder(shader, light_dir, light_pos))
        side_left = Node(transform=translate(-4, -10, 10) @ rotate(y_axis, -30) @ rotate(x_axis, 90) @ scale(1, 10, 1))
        side_left.add(WoodenCylinder(shader, light_dir, light_pos))
        side_left2 = Node(transform=translate(-15, -10, 9) @ rotate(y_axis, 60) @ rotate(x_axis, 90) @ scale(1, 15, 1))
        side_left2.add(WoodenCylinder(shader, light_dir, light_pos))
        side_right = Node(transform=translate(-4, 10, 10) @ rotate(y_axis, -30) @ rotate(x_axis, 90) @ scale(1, 10, 1))
        side_right.add(WoodenCylinder(shader, light_dir, light_pos))
        side_right2 = Node(transform=translate(-15, 10, 9) @ rotate(y_axis, 60) @ rotate(x_axis, 90) @ scale(1, 15, 1))
        side_right2.add(WoodenCylinder(shader, light_dir, light_pos))
        top = KeyFrameControlNode({0: vec(-6, 0, 14)}, {0: quaternion_from_euler(0, -20, 0)}, {0: 1})
        top_plank = Node(transform=scale(1, 10, 1) @ rotate(y_axis, 60))
        top_plank.add(WoodenCylinder(shader, light_dir, light_pos))
        arm = Node(transform=translate(-15, 0, 0) @ rotate(y_axis, 90) @ rotate(x_axis, 90) @ scale(1, 15, 1))
        arm.add(WoodenCylinder(shader, light_dir, light_pos))
        bowl = Node(transform=translate(-30, 0, 1.8) @ rotate(x_axis, 90) @ scale(5, 5, 5))
        bowl.add(*load_textured("castle/bowl.obj", shader, "castle/wood.jpg", light_dir, light_pos))
        rock = KeyFrameControlNode({0: vec(-30, 0, 5)}, {0: quaternion_from_euler(0, 0, 0)}, {0: 4})
        rock.add(*load_textured("castle/rock.obj", shader, "castle/rock.jpg", light_dir, light_pos))
        top.add(top_plank)
        top.add(arm)
        top.add(bowl)
        top.add(rock)
        self.add(base_front)
        self.add(base_left)
        self.add(base_right)
        self.add(base_back)
        self.add(left_front_wheel)
        self.add(right_front_wheel)
        self.add(left_back_wheel)
        self.add(right_back_wheel)
        self.add(side_left)
        self.add(side_left2)
        self.add(side_right)
        self.add(side_right2)
        self.add(top)

        self.top = top
        self.rock = rock

    def key_handler(self, key):
        if key == glfw.KEY_C:
            t0 = glfw.get_time()
            rotate_keys_top = {t0: quaternion_from_euler(0, -20, 0), t0 + 0.2: quaternion_from_euler(0, 90, 0)}
            translation_rock = {}
            v0 = 400
            g = 1000
            for t in np.linspace(0, 0.7, 20):
                translation_rock[t0+0.2+t] = vec(-v0*np.sin(np.pi/3)*t+(1/2)*g*t**2-30, 0, v0*np.cos(np.pi/3)*t+5)
            self.top.keyframes.rotation_kf = KeyFrames(rotate_keys_top, quaternion_slerp)
            self.rock.keyframes.translation_kf = KeyFrames(translation_rock)


# -------------- 3D resource loader -----------------------------------------
def load(file, shader):
    """ load resources from file using assimpcy, return list of ColorMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    meshes = [Mesh(shader, [m.mVertices, m.mNormals], m.mFaces)
              for m in scene.mMeshes]
    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# -------------- Function to load texture Mesh from file ----------------------------------
def load_textured(file, shader, tex_file=None, light_dir=(0, -1, -1), light_pos=(0, 1, 0)):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(tex_file=tex_file)

    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0], mesh.mNormals]
        mesh = TexturedMesh(
            shader, mat['diffuse_map'], attributes, mesh.mFaces,
            k_d=mat.get('COLOR_DIFFUSE', (0, 0, 0)),
            k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
            k_a=(0.3, 0.3, 0.3),
            s=mat.get('SHININESS', 0.),
            light_dir=light_dir,
            light_pos=light_pos
        )
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# -------------- LOAD Phong -----------------------------------------
def load_phong_mesh(file, shader, light_dir, light_pos):
    """ load resources from file using assimp, return list of ColorMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # prepare mesh nodes
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        mesh = PhongMesh(shader, [mesh.mVertices, mesh.mNormals], mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (0, 0, 0)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=(0.1, 0.1, 0.1),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir,
                         light_pos=light_pos)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# --------------- Load CSV ----------------------------------------------------
def load_csv(file, shader_texture, shader_phong, light_dir, light_pos):
    """ Load mesh with the transofrmation written in the file """
    with open(file) as f:
        csv_data = csv.reader(f, delimiter=",")
        next(csv_data)
        meshes = []
        for ligne in csv_data:
            if len(ligne) == 8:
                mesh_name, x, y, z, r_x, r_y, r_z, s = ligne
                s_x = s
                s_y = s
                s_z = s
            else:
                mesh_name, x, y, z, r_x, r_y, r_z, s_x, s_y, s_z = ligne
            place = Node(
                transform=translate(float(x), float(y), float(z))
                @ scale(float(s_x), float(s_y), float(s_z))
                @ rotate((1, 0, 0), float(r_x))
                @ rotate((0, 1, 0), float(r_y))
                @ rotate((0, 0, 1), float(r_z))
            )
            try:
                mesh = load_textured("castle/" + mesh_name, shader_texture, light_dir=light_dir, light_pos=light_pos)
            except KeyError:
                mesh = load_phong_mesh("castle/" + mesh_name, shader_phong, light_dir, light_pos)
            place.add(*mesh)
            meshes.append(place)
    return meshes


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer(1280, 720)

    # default color shader
    shader_skybox = Shader("shaders/skybox.vert", "shaders/skybox.frag")
    shader_phong = Shader("shaders/phong.vert", "shaders/phong.frag")
    shader_texture = Shader("shaders/texture.vert", "shaders/texture.frag")

    light_dir = (0, -1, 0)
    light_pos = (0, 1000, 0)

    viewer.add(FixedNode(viewer.camera, (Skybox(shader_skybox, "sky.png"),)))

    catapult = Catapult(shader_texture, light_dir, light_pos)
    catapult.transform = translate(0, 800, 35) @ rotate((0, 0, 1), -90) @ scale(5, 5, 5)

    viewer.add(Node(
        children=(
            *load_csv("castle/castle.csv", shader_texture, shader_phong, light_dir, light_pos),
            catapult,
        ),
        transform=rotate((1, 0, 0), -90)
    ))

    instructions = """
    Il s'agit du projet de graphique 3D de Nicolas Serrand, R??mi Laberg??re et Cl??ment Caffin, ??tudiants en 2A ?? Grenoble-INP ENSIMAG.
    La cam??ra peut se d??placer gr??ce aux touches ZQSD, acc??l??rer en maintenant CTRL, monter avec SPACE et descendre avec SHIFT.
    Vous pouvez lancer l'animation de la catapulte avec la touche C.
    Merci ?? vous!
    """
    print(instructions)

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
