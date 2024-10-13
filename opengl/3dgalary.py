import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os

# Initialize Pygame and OpenGL
pygame.init()
display = (1280, 720)
display = (1920,1080)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# fullscreen = True  # Set to True for fullscreen mode
# flags = DOUBLEBUF | OPENGL | (FULLSCREEN if fullscreen else 0)
# pygame.display.set_mode(display, flags)


# Set up the OpenGL environment
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glTranslatef(0.0, 0.0, -5)

# Load images
def load_textures(directory):
    textures = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                print(f"Attempting to load: {filename}")
                full_path = os.path.join(directory, filename)
                texture_surface = pygame.image.load(full_path)
                texture_data = pygame.image.tostring(texture_surface, "RGBA", 1)
                width = texture_surface.get_width()
                height = texture_surface.get_height()
                
                texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                
                textures.append(texture)
                print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return textures

# Replace with the actual path to your image directory
textures = load_textures(r'C:\Users\ALL USER\Desktop\Photobook\Pins')

# Main loop
angle = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)

    for i, texture in enumerate(textures):
        glLoadIdentity()
        glTranslatef(0, 0, -10)
        glRotatef(angle, 0, 1, 0)
        glTranslatef(i * 2 - len(textures), 0, 0)

        glBindTexture(GL_TEXTURE_2D, texture)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-1, -1, 0)
        glTexCoord2f(1, 0); glVertex3f(1, -1, 0)
        glTexCoord2f(1, 1); glVertex3f(1, 1, 0)
        glTexCoord2f(0, 1); glVertex3f(-1, 1, 0)
        glEnd()

    angle += 0.1
    pygame.display.flip()
    pygame.time.wait(10)

# Cleanup
pygame.quit()