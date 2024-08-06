import pygame
import numpy as np

class CRTShader:
    def __init__(self, screen_size):
        self.screen_size = screen_size
        self.shader_surface = pygame.Surface(screen_size)
        self.distortion = 0.1  # Adjust this value to change the curvature

    def apply(self, surface):
        width, height = self.screen_size
        pixels = pygame.surfarray.array3d(surface).astype(float)

        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        d = np.sqrt(xx*xx + yy*yy)

        # Outward bulge distortion
        d_barrel = d / (1 - self.distortion * (d**2 - 1))

        source_x = ((d_barrel/d * xx + 1) / 2 * (width - 1))
        source_y = ((d_barrel/d * yy + 1) / 2 * (height - 1))

        # Bilinear interpolation
        x0 = np.floor(source_x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(source_y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)

        wa = (x1 - source_x) * (y1 - source_y)
        wb = (x1 - source_x) * (source_y - y0)
        wc = (source_x - x0) * (y1 - source_y)
        wd = (source_x - x0) * (source_y - y0)

        distorted = (wa[:, :, np.newaxis] * pixels[x0, y0] +
                     wb[:, :, np.newaxis] * pixels[x0, y1] +
                     wc[:, :, np.newaxis] * pixels[x1, y0] +
                     wd[:, :, np.newaxis] * pixels[x1, y1])

        vignette = np.maximum(1 - d * 0.5, 0.0)
        distorted = (distorted * vignette[:,:,np.newaxis]).astype(np.uint8)

        pygame.surfarray.blit_array(self.shader_surface, distorted)

        return self.shader_surface