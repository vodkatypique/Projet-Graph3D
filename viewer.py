#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""

import sys                          # for system arguments
import os

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader
import csv

from core import RotationControlNode, Shader, Mesh, Node, KeyFrames, KeyFrameControlNode, Viewer, Texture, TexturedMesh, PhongMesh, Skybox
from transform import translate, rotate, scale, vec, quaternion, quaternion_from_euler


class Axis(Mesh):
    """ Axis object useful for debugging coordinate frames """
    def __init__(self, shader):
        pos = ((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1))
        col = ((1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1))
        super().__init__(shader, [pos, col])

    def draw(self, projection, view, model, primitives=GL.GL_LINES):
        super().draw(projection, view, model, primitives)


class SimpleTriangle(Mesh):
    """Hello triangle object"""

    def __init__(self, shader):

        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        super().__init__(shader, [position, color])


class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self, shader):
        super().__init__()
        self.add(*load('cylinder.obj', shader))  # just load cylinder from file


# -------------- 3D resource loader -----------------------------------------
def load(file, shader):
    """ load resources from file using assimpcy, return list of ColorMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
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
    except assimpcy.all.AssimpError as exception:
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
        mesh = TexturedMesh(shader, mat['diffuse_map'], attributes, mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a = (0.3, 0.3, 0.3),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir,
                         light_pos=light_pos)
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
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # prepare mesh nodes
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        mesh = PhongMesh(shader, [mesh.mVertices, mesh.mNormals], mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                        # k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         k_a = (0.1, 0.1, 0.1),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir,
                         light_pos=light_pos)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


#--------------- Load textured if it has a texture or a phong mesh -----------

def load_textured_or_phong_meshes(file, shader, light_dir, light_pos):
    return
#--------------- Load CSV ----------------------------------------------------

def load_csv(file, shader_texture, shader_phong, light_dir, light_pos):
    """ Load mesh with the transofrmation written in the file """
    with open(file) as f:
        csv_data = csv.reader(f, delimiter=",")
        next(csv_data)
        meshes = []
        for ligne in csv_data:
            mesh_name, x, y, z, R_x, R_y, R_z, S = ligne
            place = Node(transform = translate(float(x), float(y), float(z)) 
                                    @ scale(float(S), float(S), float(S)) 
                                    @ rotate((1, 0, 0), float(R_x))
                                    @ rotate((0, 1, 0), float(R_y))
                                    @ rotate((0, 0, 1), float(R_z)))
            mesh = []
            try:
                mesh = load_textured("castle/" + mesh_name, shader_texture, light_dir=light_dir, light_pos=light_pos)
            except(KeyError):
                mesh = load_phong_mesh("castle/" + mesh_name, shader_phong, light_dir, light_pos)
            place.add(*mesh)
            meshes.append(place)
    return meshes

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # default color shader
    shader = Shader("shaders/color.vert", "shaders/color.frag")
    shader_texture = Shader("shaders/texture.vert", "shaders/texture.frag")
    shader_skybox = Shader("shaders/skybox.vert", "shaders/skybox.frag")
    shader_phong = Shader("shaders/phong.vert", "shaders/phong.frag")
    
    light_dir = (0, 0, -1)
    light_pos = (0, 0, 1000)

    viewer.add(Skybox(shader_skybox, "sky.png"))
    # viewer.add(*load_textured("castle/church.FBX", 
    #                               shader_texture, light_pos=light_pos, light_dir=light_dir))
    viewer.add(*load_csv("castle/castle.csv", shader_texture, shader_phong, light_dir, light_pos))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
