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
    direction, right_vector, up_vector, screen_height, center_screen = initialize_screen_parameters(camera, args.width, args.height)

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



def compute_screen_pixels(direction,camera,image_height,image_width,screen_height,right_vector,up_vector,center_screen):

    pixels = np.zeros((image_height,image_width,len(center_screen)))
    normilzed_up_vector = np.linalg.norm(up_vector)
    normilzed_right_vector = np.linalg.norm(right_vector)
    up_edge = center_screen+normilzed_up_vector*screen_height/2
    down_edge = center_screen-normilzed_up_vector*screen_height/2
    right_edge = center_screen+normilzed_right_vector*screen_width/2
    left_edge = center_screen-normilzed_right_vector*screen_width/2

    upper_left_point = up_edge+left_edge-center_screen
    upper_right_point = up_edge+right_edge-center_screen
    lower_left_point = down_edge+left_edge-center_screen
    lower_right_point = down_edge+right_edge-center_screen

    point = upper_left_point

    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            pixels[i][j]=point
            point += (upper_right_point-upper_left_point)/image_width
        point += upper_left_point-upper_right_point
        point += (lower_left_point-upper_left_point)/image_height
    
    return pixels, normilzed_up_vector, normilzed_right_vector, up_edge, down_edge, right_edge, left_edge, upper_left_point, upper_right_point, lower_left_point, lower_right_point

def compute_ray(point1,point2):
    ray = np.zeros((2,len(point1)))
    ray[0] = point1
    ray[1] = (point2-point1)/np.linalg.norm(point2-point1)
    return ray

def compute_ray_object_intersection(ray,ob):
    if isinstance(obj, Cube):
        return compute_ray_cube_intersection(ray,obj)
    if isinstance(obj, Sphere):
        return compute_ray_sphere_intersection(ray,obj)
    if isinstance(obj, Light):
        return compute_ray_light_intersection(ray,obj)
    if isinstance(obj, InfinitePlane):
        return compute_ray_InfinitePlane_intersection(ray,obj)

def compute_ray_InfinitePlane_intersection(ray,plane):
    ray_direction = ray[1]
    ray_origin = ray[0]
    plane_point = plane.offset*plane.normal
    plane_normal = plane.normal / np.linalg.norm(plane.normal)
    denominator = np.dot(plane_normal, ray_direction)
    if np.isclose(denominator, 0):
        return None , None, None
    if t < 0:
        # The intersection point is behind the ray origin
        return None , None, None
    intersection_point = ray_origin + t * ray_direction
    return intersection_point, np.linalg.norm(intersection_point-ray[0])

def compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal):
    ray_direction = ray[1]
    ray_origin = ray[0]
    denominator = np.dot(plane_normal, ray_direction)
    if np.isclose(denominator, 0):
        return None , None, None
    if t < 0:
        # The intersection point is behind the ray origin
        return None , None, None
    intersection_point = ray_origin + t * ray_direction
    return intersection_point , np.linalg.norm(intersection_point-ray[0]), None

def compute_ray_cube_intersection(ray,cube):
    intersection = None
    x = cube.position[0]
    y = cube.position[1]
    z = cube.position[2]
    p = scale/2
    intersenction_dis = -1
    normal = None
    plane_point = cube.position+[scale/2,0,0]
    plane_normal = [1,0,0]
    cur_intersection = compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal)
    if cur_intersection!=None and cur_intersection[1]>=y-p and cur_intersection[1]<=y+p and cur_intersection[2]>=z-p and cur_intersection[2]<=z+p:
        if intersection == None or np.linalg.norm(cur_intersection-ray[0])<intersenction_dis:
            intersection = cur_intersection
            intersenction_dis = np.linalg.norm(cur_intersection-ray[0])
            normal = [1,0,0]
    
    
    plane_point = cube.position+[-scale/2,0,0]
    plane_normal = [11,0,0]
    cur_intersection = compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal)
    if cur_intersection!=None and cur_intersection[1]>=y-p and cur_intersection[1]<=y+p and cur_intersection[2]>=z-p and cur_intersection[2]<=z+p:
        if intersection == None or np.linalg.norm(cur_intersection-ray[0])<intersenction_dis:
            intersection = cur_intersection
            intersenction_dis = np.linalg.norm(cur_intersection-ray[0])
            normal = [-1,0,0]

    plane_point = cube.position+[0,p,0]
    plane_normal = [0,1,0]
    cur_intersection = compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal)
    if cur_intersection!=None and cur_intersection[0]>=x-p and cur_intersection[0]<=x+p and cur_intersection[2]>=z-p and cur_intersection[2]<=z+p:
        if intersection == None or np.linalg.norm(cur_intersection-ray[0])<intersenction_dis:
            intersection = cur_intersection
            intersenction_dis = np.linalg.norm(cur_intersection-ray[0])
            normal = [0,1,0]
    
    plane_point = cube.position+[0,-p,0]
    plane_normal = [0,-1,0]
    cur_intersection = compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal)
    if cur_intersection!=None and cur_intersection[0]>=x-p and cur_intersection[0]<=x+p and cur_intersection[2]>=z-p and cur_intersection[2]<=z+p:
        if intersection == None or np.linalg.norm(cur_intersection-ray[0])<intersenction_dis:
            intersection = cur_intersection
            intersenction_dis = np.linalg.norm(cur_intersection-ray[0])
            normal = [0,-1,0]
    
    plane_point = cube.position+[0,0,p]
    plane_normal = [0,0,1]
    cur_intersection = compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal)
    if cur_intersection!=None and cur_intersection[0]>=x-p and cur_intersection[0]<=x+p and cur_intersection[1]>=y-p and cur_intersection[1]<=y+p:
        if intersection == None or np.linalg.norm(cur_intersection-ray[0])<intersenction_dis:
            intersection = cur_intersection
            intersenction_dis = np.linalg.norm(cur_intersection-ray[0])
            normal = [0,0,1]
    
    plane_point = cube.position+[0,0,-p]
    plane_normal = [0,0,-1]
    cur_intersection = compute_ray_InfinitePlane_intersection(ray,plane_point,plane_normal)
    if cur_intersection!=None and cur_intersection[0]>=x-p and cur_intersection[0]<=x+p and cur_intersection[1]>=y-p and cur_intersection[1]<=y+p:
        if intersection == None or np.linalg.norm(cur_intersection-ray[0])<intersenction_dis:
            intersection = cur_intersection
            intersenction_dis = np.linalg.norm(cur_intersection-ray[0])
            normal = [0,0,-1]
    
    return intersection, intersenction_dis, normal

def compute_ray_Sphere_intersection(ray,sphere):
    ray_origin = ray[0]
    ray_direction = ray[1]
    sphere_center= sphere.position
    sphere_radius = sphere.radius
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Ensure direction is normalized
    oc = ray_origin - sphere_center

    a = np.dot(ray_direction, ray_direction)  # Should be 1 if normalized
    b = 2 * np.dot(ray_direction, oc)
    c = np.dot(oc, oc) - sphere_radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return None , None, None
    
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

    if  np.linalg.norm(t1-ray[0])>np.linalg.norm(t2-ray[0]):
        return t2 , np.linalg.norm(t2-ray[0]), None
    return t1, np.linalg.norm(t1-ray[0]), None

def compute_ray_lighy_intersection(ray,sphere):
    ray_origin = ray[0]
    ray_direction = ray[1]
    sphere_center= sphere.position
    sphere_radius = sphere.radius
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Ensure direction is normalized
    oc = ray_origin - sphere_center

    a = np.dot(ray_direction, ray_direction)  # Should be 1 if normalized
    b = 2 * np.dot(ray_direction, oc)
    c = np.dot(oc, oc) - sphere_radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return None , None, None
    
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

    if  np.linalg.norm(t1-ray[0])>np.linalg.norm(t2-ray[0]):
        return t2 , np.linalg.norm(t2-ray[0]), None
    return t1, np.linalg.norm(t1-ray[0]), None

def get_ray_first_collision(ray, objects):
    obj = None
    inter = -1
    dis = -1
    normal = None
    for i in range(len(objects)):
        if isinstance(objects[i], Cube) or isinstance(objects[i], Cube) or isinstance(objects[i], Cube)or isinstance(objects[i], Light):
            cur_inter, cur_dis, cur_normal = compute_ray_object_intersection(ray,objects[i])
            if cur_inter != None and (obj==None or cur_dis<dis):
                obj = objects[i]
                inter = cur_inter
                normal = cur_normal
        ##################    Take care of light ######################################

    return obj, inter, normal


