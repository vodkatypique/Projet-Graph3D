import time

import glfw
import numpy as np

from transform import normalized, lookat, perspective


class Camera:
    def __init__(
            self, window, x=0., y=0., z=0., pitch=0., yaw=0., fov=45., z_near=0.1, z_far=100000., speed=500
    ) -> None:
        self.__window = window
        self.__cursor_captured = False
        self.position = np.array([x, y, z])
        self.pitch = pitch
        self.yaw = yaw
        self.fov = fov
        self.z_near = z_near
        self.z_far = z_far
        self.speed = speed
        self.__forward = False
        self.__last_event_poll = 0
        self.__last_pos_debug = 0
        print("FORWARD: X={:.02f} Y={:.02f} Z={:.02f}".format(
            self.__get_direction()[0],
            self.__get_direction()[1],
            self.__get_direction()[2],
        ))
        print("RIGHT: X={:.02f} Y={:.02f} Z={:.02f}".format(
            self.__get_camera_right()[0],
            self.__get_camera_right()[1],
            self.__get_camera_right()[2],
        ))
        print("UP: X={:.02f} Y={:.02f} Z={:.02f}".format(
            self.__get_camera_up()[0],
            self.__get_camera_up()[1],
            self.__get_camera_up()[2],
        ))

    def __get_direction(self):
        return -normalized(np.array([
            np.cos(np.radians(self.yaw + 90)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw + 90)) * np.cos(np.radians(self.pitch))
        ]))

    def __get_camera_right(self):
        up = (0, 1, 0)
        return normalized(np.cross(self.__get_direction(), up))

    def __get_camera_up(self):
        return normalized(np.cross(self.__get_camera_right(), self.__get_direction()))

    def get_view(self):
        return lookat(self.position, self.position + self.__get_direction(), self.__get_camera_up())

    def get_projection(self, window_size):
        return perspective(self.fov, window_size[0] / window_size[1], self.z_near, self.z_far)

    def __toggle_cursor_capture(self):
        self.__cursor_captured = not self.__cursor_captured
        glfw.set_input_mode(
            self.__window, glfw.CURSOR, glfw.CURSOR_DISABLED if self.__cursor_captured else glfw.CURSOR_NORMAL
        )

    def on_key(self, key, action):
        if key == glfw.KEY_LEFT_ALT and action == glfw.PRESS:
            self.__toggle_cursor_capture()

    def process_input(self, window):
        now = time.time()
        delta_time = now - self.__last_event_poll
        self.__last_event_poll = now

        speed = self.speed * delta_time

        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            speed *= 4

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position += self.__get_direction() * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position -= self.__get_direction() * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position -= self.__get_camera_right() * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position += self.__get_camera_right() * speed

        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self.position[1] += speed
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.position[1] -= speed

        if now - self.__last_pos_debug > 0.5:
            self.__last_pos_debug = now
            print("Camera position: x={:.02f}, y={:.02f}, z={:.02f}".format(
                self.position[0], self.position[1], self.position[2]
            ))

    def on_mouse_move(self, old_position, new_position):
        if not self.__cursor_captured:
            return

        dx = new_position[0] - old_position[0]
        dy = new_position[1] - old_position[1]

        self.pitch -= dy / 5
        if self.pitch < -90:
            self.pitch = -90
        elif self.pitch > 90:
            self.pitch = 90

        self.yaw += dx / 5
        while self.yaw > 180:
            self.yaw -= 360
        while self.yaw <= -180:
            self.yaw += 360

    def capture_cursor(self):
        if not self.__cursor_captured:
            self.__toggle_cursor_capture()
