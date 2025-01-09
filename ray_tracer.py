import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer

    direction, right_vector, up_vector, screen_height, center_screen = initialize_screen_parameters

    # For each pixel:
    # 1.Shoot a ray through each pixel in the image:
    #   1.1 Discover the location of the pixel on the camera’s screen (using camera parameters)
    #   1.2 Construct a ray from the camera through that pixel.
    # 2. Check the intersection of the ray with all surfaces in the scene.
    # 3. Find the nearest intersection of the ray. This is the surface that will be seen in the image.
    # 4. Compute the color of the surface:
    #   4.1. Go over each light in the scene.
    #   4.2. Add the value it induces on the surface.
    # 5. Find out whether the light hits the surface or not:
    #   5.1. Shoot rays from the light towards the surface
    #   5.2. Find whether the ray intersects any other surfaces before the required surface - if so, the surface is occluded from the light and the light does not affect it (or partially affects it because of the shadow intensity parameter).
    # 6. Produce soft shadows, as explained below:
    #   6.1. Shoot several rays from the proximity of the light to the surface.
    #   6.2. Find out how many of them hit the required surface.

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()

####################################################

def initialize_screen_parameters(camera, image_width, image_height):
    # Calculate camera direction
    direction = np.array(camera.look_at) - np.array(camera.position)
    direction /= np.linalg.norm(direction)  # Normalize

    # Compute right and up vectors
    up_vector = np.array(camera.up_vector)
    right_vector = np.cross(direction, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(right_vector, direction)
    up_vector /= np.linalg.norm(up_vector)

    # Compute screen dimensions
    screen_height = camera.screen_width * (image_height / image_width)
    center_screen = np.array(camera.position) + direction * camera.screen_distance

    return direction, right_vector, up_vector, screen_height, center_screen

