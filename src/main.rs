use std::ops::{Add, Mul, Sub};

#[derive(Debug, Copy, Clone)]
struct V3(f32, f32, f32);

impl V3 {
    fn dot(self, other: V3) -> f32 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    fn cross(self, other: V3) -> V3 {
        V3(
            self.1 * other.2 - self.2 * other.1,
            self.2 * other.0 - self.0 * other.2,
            self.0 * other.1 - self.1 * other.0,
        )
    }

    fn normalize(self) -> V3 {
        self * (1.0 / self.len())
    }

    fn reflect(self, normal: V3) -> V3 {
        self - normal * self.dot(normal) * 2.0
    }

    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn is_unit_vector(self) -> bool {
        // TODO epsilon might be too small?
        (self.dot(self) - 1.0).abs() < f32::EPSILON
    }
}

impl Add for V3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl Add<f32> for V3 {
    type Output = Self;

    fn add(self, other: f32) -> Self {
        Self(self.0 + other, self.1 + other, self.2 + other)
    }
}

impl Sub for V3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Sub<f32> for V3 {
    type Output = Self;

    fn sub(self, other: f32) -> Self {
        Self(self.0 - other, self.1 - other, self.2 - other)
    }
}

impl Mul for V3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

impl Mul<f32> for V3 {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Self(self.0 * other, self.1 * other, self.2 * other)
    }
}

#[derive(Debug)]
struct Camera {
    origin: V3,
    x: V3,
    y: V3,
    z: V3,
    viewport_lower_left: V3,
    viewport_width: f32,
    viewport_height: f32,
}

impl Camera {
    fn new(look_from: V3, look_at: V3, aspect_ratio: f32) -> Camera {
        assert!(aspect_ratio > 1.0, "width > height only");

        let origin = look_from - look_at;
        let z = origin.normalize();
        let x = V3(0.0, 0.0, 1.0).cross(z).normalize();
        let y = z.cross(x).normalize();

        let viewport_height = 1.0;
        let viewport_width = viewport_height * aspect_ratio;
        let viewport_lower_left = origin - z - y * 0.5 * viewport_height - x * 0.5 * viewport_width;

        Camera {
            origin,
            x,
            y,
            z,
            viewport_lower_left,
            viewport_width,
            viewport_height,
        }
    }
}

enum Material {
    Diffuse { emit_color: V3, reflect_color: V3 },
    Specular { emit_color: V3, reflect_color: V3 },
}

struct Sphere {
    p: V3,
    r: f32,
    inv_r: f32,
    m: Material,
}

impl Sphere {
    fn new(p: V3, r: f32, m: Material) -> Sphere {
        Sphere {
            p,
            r,
            inv_r: 1.0 / r,
            m
        }
    }
}

fn main() {
    let width = 1920;
    let height = 1080;
    let rays_per_pixel = 1000;
    // TODO can I drop this type annotation once I use pixels?
    let pixels: Vec<u32> = Vec::with_capacity(width * height);
    let cam = Camera::new(V3(0.0, -10.0, 1.0), V3(0.0, 0.0, 0.0), width as f32/height as f32);

    let v3 = V3(1.0, 1.0, 1.0);
    let v3 = v3 + 2.0;
    println!("Hello, world! {:?}", v3);
}
