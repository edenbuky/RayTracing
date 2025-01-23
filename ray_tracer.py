import argparse
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List, Union

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


####################################################

def initialize_screen_parameters(camera, image_width, image_height):
    # Calculate camera direction
    direction = camera.look_at - camera.position
    direction /= np.linalg.norm(direction)  # Normalize

    # Compute right and up vectors
    up_vector = camera.up_vector
    right_vector = np.cross(direction, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(right_vector, direction)
    up_vector /= np.linalg.norm(up_vector)

    # Compute screen dimensions
    screen_height = camera.screen_width * (image_height / image_width)
    center_screen = np.array(camera.position) + direction * camera.screen_distance

    return direction, right_vector, up_vector, screen_height, center_screen

def compute_screen_pixels(direction, camera, image_height, image_width, screen_height, right_vector, up_vector,
                          center_screen):
    # Initialize the pixels array for storing pixel positions or colors
    pixels = np.zeros((image_height, image_width, 3))  # Assuming we're storing RGB values

    # Calculate the screen edges based on the camera's view parameters
    up_edge = center_screen + up_vector * screen_height / 2
    down_edge = center_screen - up_vector * screen_height / 2
    right_edge = center_screen + right_vector * camera.screen_width / 2
    left_edge = center_screen - right_vector * camera.screen_width / 2

    # Calculate the corner points of the screen
    upper_left_point = up_edge + left_edge - center_screen
    horizontal_step = (right_edge - left_edge) / image_width  # Keep as is
    vertical_step = -(up_edge - down_edge) / image_height  # Negative step for proper vertical orientation

    # Loop over the image dimensions
    for i in range(image_height):
        for j in range(image_width):
            # Calculate the pixel's position on the screen
            pixel_position = upper_left_point + j * horizontal_step + i * vertical_step
            pixels[i, j] = pixel_position

    # Return all relevant parameters
    return pixels

def compute_ray(point1, point2):
    ray = np.zeros((2, len(point1)))
    ray[0] = point1
    ray[1] = point2 - point1
    ray[1]  /= np.linalg.norm(ray[1])
    return ray


def compute_ray_object_intersection(ray, obj):
    if isinstance(obj, Cube):
        result = compute_ray_cube_intersection(ray, obj)
        #print(f"Cube Intersection: {result}")
        return result
    if isinstance(obj, Sphere):
        result = compute_ray_Sphere_intersection(ray, obj)
        #print(f"Sphere Intersection: {result}")
        return result
    if isinstance(obj, InfinitePlane):
        result = compute_ray_InfinitePlane_intersection(ray, obj)
        #print(f"Plane Intersection: {result}")
        return result

def compute_ray_InfinitePlane_intersection(ray, plane):
    ray_origin = ray[0]
    ray_direction = ray[1]

    # Calculate the denominator of the intersection equation
    denominator = np.dot(ray_direction, plane.normal)

    # If the denominator is close to 0, the ray is parallel to the plane
    if np.isclose(denominator, 0):
        return None, None, None

    # Calculate the distance t to the intersection point
    t = np.dot(plane.plane_point - ray_origin, plane.normal) / denominator

    # If t < 0, the intersection is behind the ray's origin
    if t < 0:
        return None, None, None

    # Calculate the intersection point
    intersection_point = ray_origin + t * ray_direction

    return intersection_point, t, plane.normal



def compute_ray_cube_intersection(ray: List[np.ndarray], cube: 'Cube') -> Tuple[
    Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:

    ray_origin, ray_direction = ray

    # Ensure ray direction is normalized
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Calculate cube bounds
    half_scale = cube.scale / 2
    min_bound = cube.position - half_scale
    max_bound = cube.position + half_scale

    # Calculate inverses and handle division by zero atomically
    dir_inv = np.zeros_like(ray_direction)
    mask = np.abs(ray_direction) > 1e-10
    dir_inv[mask] = 1.0 / ray_direction[mask]
    # Components where ray_direction is effectively zero remain zero in dir_inv

    # Calculate intersection with all slabs
    t1 = (min_bound - ray_origin) * dir_inv
    t2 = (max_bound - ray_origin) * dir_inv

    # For directions close to zero, set appropriate t values
    # If ray origin is inside the slab, use -inf and inf
    # If ray origin is before the slab, use inf for both
    # If ray origin is after the slab, use -inf for both
    zero_dirs = ~mask
    if np.any(zero_dirs):
        for i in np.where(zero_dirs)[0]:
            if min_bound[i] <= ray_origin[i] <= max_bound[i]:
                t1[i] = -np.inf
                t2[i] = np.inf
            else:
                if ray_origin[i] < min_bound[i]:
                    t1[i] = np.inf
                    t2[i] = np.inf
                else:  # ray_origin[i] > max_bound[i]
                    t1[i] = -np.inf
                    t2[i] = -np.inf

    # Get entry and exit points
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)

    # Find the largest entry and smallest exit
    t_near = np.max(t_min)
    t_far = np.min(t_max)

    # Check if there's a valid intersection
    if t_near > t_far or t_far < 0:
        return None, None, None

    # If t_near is negative, ray starts inside cube
    t_hit = t_near if t_near >= 0 else t_far

    # Calculate intersection point
    intersection_point = ray_origin + ray_direction * t_hit

    # Calculate normal (using the face that was hit)
    eps = 1e-10  # Small epsilon for numerical stability
    normal = np.zeros(3)

    # Find which face was hit by comparing intersection point with bounds
    for i in range(3):
        if abs(intersection_point[i] - min_bound[i]) < eps:
            normal[i] = -1
            break
        elif abs(intersection_point[i] - max_bound[i]) < eps:
            normal[i] = 1
            break

    # Ensure normal is normalized
    normal = normal / np.linalg.norm(normal)

    return intersection_point, t_hit, normal


#
# def compute_ray_cube_intersection(ray: List[np.ndarray], cube: 'Cube') -> Tuple[
#     Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
#
#     ray_origin, ray_direction = ray
#
#     # Ensure ray direction is normalized
#     ray_direction = ray_direction / np.linalg.norm(ray_direction)
#
#     # Calculate cube bounds
#     half_scale = cube.scale / 2
#     min_bound = cube.position - half_scale
#     max_bound = cube.position + half_scale
#
#     # Handle division by zero in ray direction components
#     dir_inv = np.where(
#         np.abs(ray_direction) < 1e-10,
#         np.inf,
#         1.0 / ray_direction
#     )
#
#     # Calculate intersection with all slabs
#     t1 = (min_bound - ray_origin) * dir_inv
#     t2 = (max_bound - ray_origin) * dir_inv
#
#     # Get entry and exit points
#     t_min = np.minimum(t1, t2)
#     t_max = np.maximum(t1, t2)
#
#     # Find the largest entry and smallest exit
#     t_near = np.max(t_min)
#     t_far = np.min(t_max)
#
#     # Check if there's a valid intersection
#     if t_near > t_far or t_far < 0:
#         return None, None, None
#
#     # If t_near is negative, ray starts inside cube
#     t_hit = t_near if t_near >= 0 else t_far
#
#     # Calculate intersection point
#     intersection_point = ray_origin + ray_direction * t_hit
#
#     # Calculate normal (using the face that was hit)
#     eps = 1e-10  # Small epsilon for numerical stability
#     normal = np.zeros(3)
#
#     # Find which face was hit by comparing intersection point with bounds
#     for i in range(3):
#         if abs(intersection_point[i] - min_bound[i]) < eps:
#             normal[i] = -1
#             break
#         elif abs(intersection_point[i] - max_bound[i]) < eps:
#             normal[i] = 1
#             break
#
#     # Ensure normal is normalized
#     normal = normal / np.linalg.norm(normal)
#
#     return intersection_point, t_hit, normal


def compute_ray_Sphere_intersection(ray, sphere, epsilon=1e-5):
    ray_origin = ray[0]
    ray_direction = ray[1]
    sphere_center = sphere.position
    sphere_radius = sphere.radius

    # Vector from the sphere center to the ray origin
    oc = ray_origin - sphere_center

    # Coefficients of the quadratic equation
    A = np.dot(ray_direction, ray_direction)  # Should be 1 if direction is normalized
    B = 2.0 * np.dot(ray_direction, oc)
    C = np.dot(oc, oc) - sphere_radius ** 2

    # Discriminant
    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        # No intersection
        return None, None, None

    # Compute the two roots of the quadratic equation
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B - sqrt_discriminant) / (2.0 * A)
    t2 = (-B + sqrt_discriminant) / (2.0 * A)

    # Choose the closest positive t
    if t1 > epsilon:
        t = t1
    elif t2 > epsilon:
        t = t2
    else:
        # Both intersections are behind the ray origin
        return None, None, None

    # Compute the intersection point
    intersection_point = ray_origin + t * ray_direction

    # Compute the normal at the intersection point
    normal = (intersection_point - sphere_center) / sphere_radius  # Normalize the normal
    normal /= np.linalg.norm(normal)
    return intersection_point, t, normal


def get_ray_first_collision(ray, objects):
    obj = None
    inter = -1
    dis = -1
    normal = None
    for i in range(len(objects)):
        if isinstance(objects[i], Cube) or isinstance(objects[i], Sphere) or isinstance(objects[i], InfinitePlane):
            cur_inter, cur_dis, cur_normal = compute_ray_object_intersection(ray, objects[i])
            if cur_inter is not None and (obj is None or cur_dis < dis):
                # if isinstance(objects[i], Sphere):
                #    print("Sphere")
                obj = objects[i]
                inter = cur_inter
                dis = cur_dis
                normal = cur_normal

    return obj, inter, normal


# def soft_shadows(light, intersection_point, objects, shadow_ray, intersected_obj, sh_rays, epsilon=1e-15):
#     N = int(sh_rays)  # Number of shadow rays (total rays: N^2)
#     total_rays = N * N
#
#     # Calculate the direction of the main light ray
#     light_direction = shadow_ray[1]
#     light_direction /= np.linalg.norm(light_direction)
#
#     # Compute perpendicular vectors to the light direction
#     up_candidate = np.array([0, 1, 0]) if not np.allclose(light_direction, [0, 1, 0]) else np.array([1, 0, 0])
#     right_vector = np.cross(light_direction, up_candidate)
#     right_vector /= np.linalg.norm(right_vector)
#     up_vector = np.cross(right_vector, light_direction)
#
#     # Calculate the size of each grid cell
#     cell_size = (2 * light.radius) / N  # Size of each grid cell (full radius)
#     rays_hit = 0
#
#     for i in range(N):
#         for j in range(N):
#             # Define the current grid cell boundaries
#             cell_min_x = -light.radius + i * cell_size
#             cell_min_y = -light.radius + j * cell_size
#             cell_max_x = cell_min_x + cell_size
#             cell_max_y = cell_min_y + cell_size
#
#             # Select a random point within the current grid cell
#             random_x = np.random.uniform(cell_min_x, cell_max_x)
#             random_y = np.random.uniform(cell_min_y, cell_max_y)
#             random_offset = random_x * right_vector + random_y * up_vector
#
#             # Compute the sample point on the light
#             light_sample_point = light.position + random_offset
#
#             # Create a ray from the sample point on the light to the intersection point
#             sample_ray = compute_ray(light_sample_point, intersection_point)
#
#             # Check for occlusion
#             blocking_obj, blocking_point, _ = get_ray_first_collision(sample_ray, objects)
#             if blocking_obj is None or (
#                 blocking_obj is intersected_obj and np.linalg.norm(blocking_point - intersection_point) < epsilon
#             ):
#                 rays_hit += 1
#
#     # Compute the final light intensity
#     light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (rays_hit / total_rays)
#     return light_intensity
def soft_shadows(light, intersection_point, objects, shadow_ray, intersected_obj, sh_rays, epsilon=1e-10):
    N = int(sh_rays)   # Number of shadow rays (N^2 total rays)
    total_rays = N * N
    # Define the light's perpendicular plane
    light_direction = shadow_ray[1]  # Use the provided normalized direction
    light_direction /= np.linalg.norm(light_direction)

    # Calculate right_vector and up_vector
    up_candidate = np.array([0, 1, 0]) if not np.allclose(light_direction, [0, 1, 0]) else np.array([1, 0, 0])
    right_vector = np.cross(light_direction, up_candidate)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(right_vector, light_direction)

    # Define the rectangle grid around the light
    half_radius = light.radius / 2
    cell_size = 2 * half_radius / N  # Precompute the size of each cell
    offsets = np.linspace(-half_radius, half_radius, N)  # Grid offsets

    rays_hit = 0

    # Loop over the grid
    for i in range(N):
        for j in range(N):
            # Randomize the offset within the current cell
            random_x = np.random.uniform(offsets[i], offsets[i] + cell_size)
            random_y = np.random.uniform(offsets[j], offsets[j] + cell_size)
            random_offset = random_x * right_vector + random_y * up_vector

            # Compute the randomized sample point
            light_sample_point = light.position + random_offset

            # Cast a shadow ray
            sample_ray = compute_ray(light_sample_point, intersection_point)

            # Check for occlusion
            blocking_obj, blocking_point, _ = get_ray_first_collision(sample_ray, objects)
            if (
                    blocking_obj is not None and
                    (blocking_obj is not intersected_obj or
                     np.linalg.norm(blocking_point - intersection_point) > epsilon)
            ):
                continue
            rays_hit += 1

    # Compute light intensity
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (rays_hit / total_rays)
    return light_intensity


def calculate_reflection(ray, hit_point, hit_normal, hit_object, objects, materials, lights, recursion_depth, scene_settings):
    material = materials[hit_object.material_index - 1]
    # If we have reached maximum depth or there is no reflection
    if recursion_depth >= scene_settings.max_recursions or np.linalg.norm(material.reflection_color) == 0:
        return calculate_shading(ray, hit_point, hit_normal, hit_object, material, lights, objects, scene_settings)
    # Calculating the direction of the reflected beam
    reflection_direction = ray[1] - 2 * np.dot(ray[1], hit_normal) * hit_normal
    reflection_direction /= np.linalg.norm(reflection_direction)
    reflection_ray = [hit_point + 1e-5 * reflection_direction, reflection_direction]
    if isinstance(hit_object, Cube):
        print(f"Hit_object {hit_object}: Normal: {hit_normal}, Intersection_point: {hit_point},Ray origin:{ray[0]}, Ray direction: {ray[1]}, Reflection direction: {reflection_direction} ")
    # Collision check on new object
    obj, next_point, next_normal = get_ray_first_collision(reflection_ray, objects)
    # If there is no hit, return background color
    if obj is None:
        return scene_settings.background_color * material.reflection_color

    # Recursive call for the next object
    reflected_color = calculate_reflection(
        reflection_ray, next_point, next_normal, obj, objects, materials, lights, recursion_depth + 1, scene_settings
    )

    # Calculate the shading color for the current object
    shading_color = calculate_shading(ray, next_point, next_normal, obj, materials[obj.material_index - 1], lights, objects, scene_settings)

    # Color combination
    final_color = (shading_color + reflected_color) * material.reflection_color

    max_component = max(1, np.max(final_color))
    final_color = final_color / max_component

    return np.clip(final_color,0,1)



# def calculate_reflection_work(ray, intersection_point, normal, object, objects, materials, lights,
#                          recursion_depth, scene_settings):
#     material = materials[object.material_index - 1]
#
#     if recursion_depth >= scene_settings.max_recursions:
#         return np.zeros(3)  # Stop recursion if max depth is reached
#
#
#     # Calculate the direction of the reflection ray
#     reflection_direction = ray[1] - 2 * np.dot(ray[1], normal) * normal
#     reflection_direction /= np.linalg.norm(reflection_direction)  # Normalize the reflection direction
#
#     # Generate the reflection ray
#     reflection_ray = [intersection_point + 1e-5 * reflection_direction, reflection_direction]
#
#     # Check for the next object the reflection ray hits
#     next_obj, next_point, next_normal = get_ray_first_collision(reflection_ray, objects)
#
#     if next_obj is None:
#         # If no object is hit, use the background color
#         return scene_settings.background_color * material.reflection_color
#     else:
#         # Compute reflection recursively
#         # Add shading from the new intersection
#         shading_color = calculate_shading(ray, next_point, next_normal, next_obj, material, lights, objects,
#                                           scene_settings)
#         reflection_color = calculate_reflection(
#             reflection_ray, next_point, next_normal, next_obj, objects, materials, lights,
#             recursion_depth + 1, scene_settings
#         )
#
#     # Multiply the reflection color by the material's reflection coefficient
#     return reflection_color * material.reflection_color

def calculate_shading(ray, intersection_point, normal, obj, obj_material, lights, objects, scene_settings):
    # Initialize the final color as black
    final_color = np.zeros(3)
    view_direction = (ray[0] - intersection_point)
    view_direction /= np.linalg.norm(view_direction)

    for light in lights:
        # Compute the ray from the intersection point to the light
        shadow_ray = compute_ray(intersection_point, light.position)

        # Calculate the light intensity at the intersection point using soft_shadows
        light_intensity = soft_shadows(light, intersection_point, objects, shadow_ray, obj, scene_settings.root_number_shadow_rays)

        # Compute diffuse lighting
        light_direction = shadow_ray[1]  # Normalized direction of the ray
        diffuse_intensity = max(0, np.dot(normal, light_direction))
        diffuse_color = diffuse_intensity * np.array(light.color) * np.array(obj_material.diffuse_color)

        # Compute specular lighting
        reflection_direction = 2 * np.dot(normal, light_direction) * normal - light_direction
        specular_intensity = max(0, np.dot(reflection_direction, view_direction)) ** obj_material.shininess
        specular_color = specular_intensity * np.array(light.color) * np.array(obj_material.specular_color) * light.specular_intensity

        # Accumulate the lighting contributions
        final_color += light_intensity * (diffuse_color + specular_color)


    # Ensure values are within range [0, 1]
    return final_color

#############################################

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    materials = [o for o in objects if isinstance(o, Material)]
    lights = [o for o in objects if isinstance(o, Light)]
    physical_objects = [o for o in objects if not isinstance(o, (Light, Material))]

    # TODO: Implement the ray tracer
    direction, right_vector, up_vector, screen_height, center_screen = initialize_screen_parameters(camera, args.width,
                                                                                                    args.height)
    # Compute all pixel positions on the screen
    pixels = compute_screen_pixels(direction, camera, args.height, args.width,
                                                                 screen_height, right_vector, up_vector, center_screen)

    # Initialize the image array
    image_array = np.zeros((args.height, args.width, 3))
    rev = len(image_array[0])-1
    # Loop over each pixel
    for i in range(args.height):
        print(i)
        for j in range(args.width):
            # Get the pixel position
            pixel_position = pixels[i, j]

            # Construct a ray from the camera through that pixel
            ray = compute_ray(camera.position, pixel_position)

            # Check the intersection of the ray with all surfaces in the scene
            obj, intersection_point, normal = get_ray_first_collision(ray, physical_objects)

            if obj is None:
                # No intersection: set the background color
                image_array[i, j] = scene_settings.background_color
                continue
            obj_material = materials[obj.material_index -1]
            # Compute the color of the surface
            color = calculate_shading(ray, intersection_point, normal,obj, obj_material, lights, physical_objects, scene_settings)

            # Add reflection color
            if np.linalg.norm(obj_material.reflection_color) > 0:
                reflection_color = calculate_reflection(
                    ray=ray,
                    hit_point=intersection_point,
                    hit_normal=normal,
                    hit_object= obj,
                    objects=physical_objects,
                    materials=materials,
                    lights = lights,
                    recursion_depth=1,
                    scene_settings=scene_settings
                )
                color += reflection_color

            # Assign the calculated color to the pixel
            image_array[i, rev -j] = np.clip(color,0,1)


    image_array = (image_array * 255).astype(np.uint8)
    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
