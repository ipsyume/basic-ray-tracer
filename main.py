# main.py
import numpy as np
from PIL import Image
import os

# Ensure output directory exists
os.makedirs("images", exist_ok=True)

# ---------------- Math Utilities ----------------
def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def dot(a, b):
    return np.dot(a, b)

def reflect(I, N):
    # Reflect incident vector I around normal N
    return I - 2 * dot(I, N) * N

# ---------------- Scene Objects ----------------
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.5):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.color = np.array(color, dtype=float)
        self.specular = specular
        self.reflective = reflective

    def intersect(self, origin, direction):
        oc = origin - self.center
        a = dot(direction, direction)
        b = 2.0 * dot(oc, direction)
        c = dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.inf
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        eps = 1e-4
        ts = [t for t in (t1, t2) if t > eps]
        return min(ts) if ts else np.inf

    def normal_at(self, p):
        return normalize(p - self.center)

class Plane:
    def __init__(self, point, normal, color, specular=10, reflective=0.2, checkerboard=False):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.color = np.array(color, dtype=float)
        self.specular = specular
        self.reflective = reflective
        self.checkerboard = checkerboard

    def intersect(self, origin, direction):
        denom = dot(self.normal, direction)
        if abs(denom) < 1e-6:
            return np.inf
        t = dot(self.point - origin, self.normal) / denom
        return t if t > 1e-4 else np.inf

    def normal_at(self, p):
        return self.normal

    def get_color(self, point):
        if not self.checkerboard:
            return self.color
        # Simple XZ checker pattern
        scale = 1.0
        x, z = point[0], point[2]
        check = (int(np.floor(x * scale)) + int(np.floor(z * scale))) % 2
        return self.color if check == 0 else np.array([30, 30, 30])

# ---------------- Light ----------------
class Light:
    def __init__(self, position, intensity):
        self.position = np.array(position, dtype=float)
        self.intensity = float(intensity)

# ---------------- Shading ----------------
def compute_lighting(point, normal, view_dir, specular, objects, lights):
    # Ambient
    intensity = 0.05
    for light in lights:
        to_light = normalize(light.position - point)
        light_distance = np.linalg.norm(light.position - point)

        # Shadow ray
        shadow_origin = point + normal * 1e-3
        shadowed = False
        for obj in objects:
            t = obj.intersect(shadow_origin, to_light)
            if t < light_distance:
                shadowed = True
                break
        if shadowed:
            continue

        # Diffuse
        diff = max(dot(normal, to_light), 0.0)
        intensity += light.intensity * diff

        # Specular (Blinn-Phong)
        if specular > 0:
            half_vec = normalize(to_light + view_dir)
            spec = max(dot(normal, half_vec), 0.0) ** specular
            intensity += light.intensity * spec * 0.3
    return intensity

# ---------------- Ray Tracing ----------------
def trace_ray(origin, direction, objects, lights, depth=3):
    closest_t = np.inf
    hit_obj = None
    for obj in objects:
        t = obj.intersect(origin, direction)
        if t < closest_t:
            closest_t = t
            hit_obj = obj

    if hit_obj is None:
        return np.array([30, 30, 40])  # background color

    hit_point = origin + closest_t * direction
    normal = hit_obj.normal_at(hit_point)
    view_dir = normalize(-direction)

    # Base color (checkerboard for plane)
    if isinstance(hit_obj, Plane):
        base_color = hit_obj.get_color(hit_point)
    else:
        base_color = hit_obj.color

    # Local shading
    lighting = compute_lighting(hit_point, normal, view_dir, hit_obj.specular, objects, lights)
    local_color = base_color * lighting

    # Reflections
    r = hit_obj.reflective
    if depth <= 0 or r <= 0:
        return np.clip(local_color, 0, 255)

    reflect_dir = normalize(reflect(direction, normal))
    reflect_origin = hit_point + normal * 1e-3
    reflect_color = trace_ray(reflect_origin, reflect_dir, objects, lights, depth - 1)

    final_color = (1 - r) * local_color + r * reflect_color
    return np.clip(final_color, 0, 255)

# ---------------- Rendering ----------------
def render(width=640, height=480, samples_per_pixel=4):
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    camera = np.array([0.0, 0.0, -1.0])
    viewport_size = 1.0
    projection_plane_z = 1.0

    def canvas_to_viewport(x, y):
        return np.array([
            x * viewport_size / width,
            y * viewport_size / height,
            projection_plane_z
        ])

    # Scene definition
    objects = [
        Sphere([0, -0.2, 3], 0.8, [220, 60, 60], specular=100, reflective=0.4),
        Sphere([1.5, 0.2, 4], 0.9, [60, 120, 240], specular=50, reflective=0.3),
        Sphere([-1.8, 0.0, 4.5], 1.0, [60, 200, 120], specular=30, reflective=0.2),
        Plane([0, -1, 0], [0, 1, 0], [210, 200, 180], specular=10, reflective=0.2, checkerboard=True),
    ]
    lights = [
        Light([2, 3, -1], 1.4),
        Light([-3, 5, -2], 0.8),
    ]

    # Anti-aliasing via supersampling (uniform grid per pixel)
    for cx in range(-width // 2, width // 2):
        for cy in range(-height // 2, height // 2):
            color = np.zeros(3)
            for sx in range(samples_per_pixel):
                for sy in range(samples_per_pixel):
                    dx = (sx + 0.5) / samples_per_pixel
                    dy = (sy + 0.5) / samples_per_pixel
                    x = cx + dx
                    y = cy + dy
                    direction = normalize(canvas_to_viewport(x, y))
                    color += trace_ray(camera, direction, objects, lights, depth=3)
            color /= (samples_per_pixel ** 2)

            px = cx + width // 2
            py = height // 2 - cy - 1
            pixels[px, py] = tuple(color.astype(np.uint8))

    out_path = "images/final_ray_tracer.png"
    image.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    render(640, 480, samples_per_pixel=3)
