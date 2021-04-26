# Python built-in modules
import time                         # for frames statistics
from itertools import cycle         # allows easy circular choice list
from bisect import bisect_left      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
from OpenGL.GLUT import *
# our transform functions
from camera import Camera
from transform import identity, rotate, lerp, quaternion_slerp, quaternion_matrix, translate, scale

from PIL import Image


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            sys.exit(1)
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                sys.exit(1)

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

    def loc(self, loc_name):
        return GL.glGetUniformLocation(self.glid, loc_name)

    def setup_camera(self, view):
        # world camera position for Phong illumination specular component
        w_camera_position = np.linalg.inv(view)[:, 3]
        GL.glUniform3fv(self.loc('w_camera_position'), 1, w_camera_position)

    def setup_material(self, k_a, k_d, k_s, s):
        GL.glUniform3fv(self.loc('k_a'), 1, k_a)
        GL.glUniform3fv(self.loc('k_d'), 1, k_d)
        GL.glUniform3fv(self.loc('k_s'), 1, k_s)
        GL.glUniform1f(self.loc('s'), max(s, 0.001))

    def setup_light(self, light_dir, light_pos):
        GL.glUniform3fv(self.loc('light_dir'), 1, light_dir)
        GL.glUniform3fv(self.loc('light_pos'), 1, light_pos)


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers.append(GL.glGenBuffers(1))
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


# ------------  Mesh is a core drawable, can be basis for most objects --------
class Mesh:
    """ Basic mesh class with attributes passed as constructor arguments """
    def __init__(self, shader, attributes, index=None):
        self.shader = shader
        names = ['view', 'projection', 'model']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(primitives)


# ------------  Node is the core drawable for hierarchical scene graphs -------
class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, children=(), transform=identity()):
        self.transform = transform
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model):
        """ Recursive draw, passing down updated model matrix. """
        for child in self.children:
            child.draw(projection, view, model @ self.transform)

    def key_handler(self, key):
        """ Dispatch keyboard events to children """
        for child in self.children:
            if hasattr(child, 'key_handler'):
                child.key_handler(key)


class FixedNode(Node):
    """ Scene graph node that follow camera movement to never move on the screen """
    def __init__(self, camera: Camera, children=()):
        super().__init__(children, translate(camera.position))
        self.camera = camera

    def draw(self, projection, view, model):
        self.transform = translate(self.camera.position)
        super().draw(projection, view, model)


class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time_sec):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time_sec <= self.times[0]:
            return self.values[0]
        elif time_sec >= self.times[-1]:
            return self.values[-1]
        # 2. search for closest index entry in self.times, using bisect_left function
        closest_index = bisect_left(self.times, time_sec) - 1
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        fraction = (time_sec - self.times[closest_index]) / (self.times[closest_index + 1] - self.times[closest_index])
        return self.interpolate(self.values[closest_index], self.values[closest_index + 1], fraction)


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translation_kf = KeyFrames(translate_keys)
        self.rotation_kf = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale_kf = KeyFrames(scale_keys)

    def value(self, time_sec):
        """ Compute each component's interpolation and compose TRS matrix """
        translation_value = self.translation_kf.value(time_sec)
        rotation_value = self.rotation_kf.value(time_sec)
        scale_value = self.scale_kf.value(time_sec)
        return translate(translation_value) @ quaternion_matrix(rotation_value) @ scale(scale_value)


class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        super().__init__()
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model)


# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, tex_file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            tex = np.asarray(Image.open(tex_file).convert('RGBA'))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (tex_file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % tex_file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

    def __iter__(self):
        return [self].__iter__()


# -------------- Texture Mesh class ----------------------------------
class TexturedMesh(Mesh):
    """ Texture Mesh class """

    def __init__(self, shader, texture, attributes, index, second_texture=None,
                 light_dir=(0, -1, -1), light_pos=(0, 1, 0), k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):
        super().__init__(shader, attributes, index)
        self.texture = texture
        self.second_texture = second_texture
        self.light_dir = light_dir
        self.light_pos = light_pos
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s

        names = ['diffuse_map', 'diffuse_map_2']
        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # setup shader parameters
        self.shader.setup_light(self.light_dir, self.light_pos)
        self.shader.setup_material(self.k_a, self.k_d, self.k_s, max(self.s, 0.001))
        self.shader.setup_camera(view)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)

        if self.second_texture:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.second_texture.glid)
            GL.glUniform1i(self.loc['diffuse_map_2'], 1)

        super().draw(projection, view, model, primitives)


# -------------- Phong rendered Mesh class -----------------------------------
class PhongMesh(Mesh):
    """ Mesh with Phong illumination """

    def __init__(self, shader, attributes, index=None,
                 light_dir=(0, -1, 0), light_pos=(0, 1, 0),
                 k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):
        super().__init__(shader, attributes, index)
        self.light_dir = light_dir
        self.light_pos = light_pos
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s

        # retrieve OpenGL locations of shader variables at initialization
        names = ['light_dir', 'light_pos', 'k_a', 's', 'k_s', 'k_d', 'w_camera_position']

        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):

        GL.glUseProgram(self.shader.glid)

        # setup shader parameters
        self.shader.setup_light(self.light_dir, self.light_pos)
        self.shader.setup_material(self.k_a, self.k_d, self.k_s, max(self.s, 0.001))
        self.shader.setup_camera(view)

        super().draw(projection, view, model, primitives)
    
# ----------------------------- Skybox ---------------------------
class Skybox(Mesh):
    """ Skybox object """

    def __init__(self, shader, tex_file):
        # Comments assume default starting position: facing -Z
        vertices = 40000 * np.array((
            (-1, -1, -1),   # 0 - bottom far left
            (1, -1, -1),    # 1 - bottom far right
            (1, 1, -1),     # 2 - top far right
            (-1, 1, -1),    # 3 - top far left
            (-1, -1, 1),    # 4 - bottom near left
            (1, -1, 1),     # 5 - bottom near right
            (1, 1, 1),      # 6 - top near right
            (-1, 1, 1)      # 7 - top near left
        ), np.float32)
        faces = np.array((
            (0, 1, 2), (0, 2, 3),   # far face
            (5, 1, 0), (4, 5, 0),   # bottom face
            (0, 3, 7), (0, 7, 4),   # left face
            (3, 2, 6), (3, 6, 7),   # top face
            (6, 2, 1), (5, 6, 1),   # back face
            (6, 5, 4), (7, 6, 4)    # near face
        ), np.uint32)
        super().__init__(shader, [vertices], faces)

        names = ['diffuse_map', 'w_camera_position']
        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)

        self.tex_file = tex_file

        self.texture_glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.texture_glid)

            for i in range(6):
                image = Image.open("sky/sky{}.png".format(i)).transpose(Image.FLIP_LEFT_RIGHT).resize((900, 900))
                tex = image.tobytes()
                GL.glTexImage2D(
                    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGBA, 900,
                    900, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex
                )

            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
            message = 'Loaded texture %s\t'
            print(message % tex_file)
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % tex_file)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.texture_glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        super().draw(projection, view, model, primitives)


# -------------- Example texture plane class ----------------------------------
class TexturedPlane(TexturedMesh):
    """ Simple first textured object """

    def __init__(self, tex_file, shader, light_dir, light_pos):

        vertices = 100 * np.array(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
        faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
        texture_coord = 100 * np.array(
            ((-1, -1), (1, -1), (1, 1), (-1, 1)), np.float32)
        normals = np.array(((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)), np.float32)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.tex_file = tex_file
        self.texture = Texture(tex_file, self.wrap_mode, *self.filter_mode)

        super().__init__(shader, self.texture, [vertices, texture_coord, normals], faces,
                         light_dir=light_dir, light_pos=light_pos,
                         k_a=(0.3, 0.3, 0.3), k_s=(0.5, 0.5, 0.5))


# ------------  Viewer class & window management ------------------------------
def on_size(win, _width, _height):
    """ window size update => update viewport to new framebuffer size """
    GL.glViewport(0, 0, *glfw.get_framebuffer_size(win))


class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        super().__init__()

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)

        # initialize camera
        self.camera = Camera(self.win, y=100, z=100)
        self.mouse = (0, 0)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_mouse_button_callback(self.win, self.on_mouse_click)
        glfw.set_window_size_callback(self.win, on_size)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_CULL_FACE)   # backface culling enabled (TP2)
        GL.glEnable(GL.GL_DEPTH_TEST)  # depth test now enabled (TP2)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def run(self):
        """ Main render loop for this OpenGL window """
        last_debug_time = time.time()
        last_frame_time = 0.
        frames = 0
        while not glfw.window_should_close(self.win):

            now = time.time() * 1000
            rendered = False
            if now - last_frame_time >= 1000 / 144:     # Target 144FPS
                # clear draw buffer and depth buffer (<-TP2)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                win_size = glfw.get_window_size(self.win)
                view = self.camera.get_view()
                projection = self.camera.get_projection(win_size)

                # draw our scene objects
                self.draw(projection, view, identity())

                # flush render commands, and swap draw buffers
                glfw.swap_buffers(self.win)

                # Update frame statistics
                rendered = True
                frames += 1

            # Poll for and process events
            self.camera.process_input(self.win)
            glfw.poll_events()

            if time.time() - last_debug_time >= 1:
                glfw.set_window_title(self.win, "Viewer ({:.02f} FPS)".format(frames))
                last_debug_time = time.time()
                frames = 0

            if rendered:
                last_frame_time = time.time() * 1000

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_F:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_BACKSPACE:
                glfw.set_time(0)

            self.camera.on_key(key, action)

            # call Node.key_handler which calls key_handlers for all drawables
            self.key_handler(key)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        self.camera.on_mouse_move(old, self.mouse)

    def on_mouse_click(self, _win, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.camera.capture_cursor()
