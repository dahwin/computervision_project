import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# vertices = (
#     (1, -1, -1),
#     (1, 1, -1),
#     (-1, 1, -1),
#     (-1, -1, -1),
#     (1, -1, 1),
#     (1, 1, 1),
#     (-1, -1, 1),
#     (-1, 1, 1)
# )

# edges = (
#     (0, 1),
#     (1, 2),
#     (2, 3),
#     (3, 0),
#     (4, 5),
#     (5, 6),
#     (6, 7),
#     (7, 4),
#     (0, 4),
#     (1, 5),
#     (2, 6),
#     (3, 7)
# )

# colors = (
#     (1, 0, 0),
#     (0, 1, 0),
#     (0, 0, 1),
#     (1, 1, 0),
#     (1, 0, 1),
#     (0, 1, 1),
#     (1, 1, 1),
#     (0.5, 0.5, 0.5)
# )

# def draw_cube():
#     glBegin(GL_QUADS)
#     for surface in ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6)):
#         for vertex in surface:
#             glColor3fv(colors[vertex])
#             glVertex3fv(vertices[vertex])
#     glEnd()

#     glBegin(GL_LINES)
#     for edge in edges:
#         for vertex in edge:
#             glVertex3fv(vertices[vertex])
#     glEnd()

# def main():
#     pygame.init()
#     display = (800, 600)
#     pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)  # Enable window resizing
#     pygame.display.set_caption("Dahyun window")  # Set window title

#     # Load and set custom window icon
#     icon = pygame.image.load('dahyun.png')
#     pygame.display.set_icon(icon)
    
#     gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
#     glTranslatef(0.0, 0.0, -5)
#     glEnable(GL_DEPTH_TEST)

#     clock = pygame.time.Clock()
#     angle = 0

#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#             elif event.type == VIDEORESIZE:
#                 glViewport(0, 0, event.w, event.h)
#                 glMatrixMode(GL_PROJECTION)
#                 glLoadIdentity()
#                 gluPerspective(45, event.w / event.h, 0.1, 50.0)
#                 glMatrixMode(GL_MODELVIEW)
#                 glLoadIdentity()
#                 glTranslatef(0.0, 0.0, -5)

#         glRotatef(1, 3, 1, 1)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         draw_cube()
#         pygame.display.flip()
#         clock.tick(60)

# if __name__ == "__main__":
#     main()

from OpenGL import GL,GLU
x = dir(GLU)
print("\n".join(x))
print(len(x))