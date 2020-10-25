use pico_args::Arguments;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{
    cell::Cell,
    cmp,
    f32::consts::PI,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Sub},
    time::Instant,
};

const TOLERANCE: f32 = 0.0001;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
type LaneF32 = F32x8;

#[derive(Debug, Copy, Clone)]
struct V3(f32, f32, f32);

#[derive(Debug, Copy, Clone)]
struct F32x8(__m256);

impl From<f32> for F32x8 {
    fn from(x: f32) -> Self {
        Self(_mm256_set1_ps(x))
    }
}

impl Add for F32x8 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(_mm256_add_ps(self.0, other.0))
    }
}

impl AddAssign for F32x8 {
    fn add_assign(&mut self, other: Self) {
        self.0 = _mm256_add_ps(self.0, other.0)
    }
}

impl Div for F32x8 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(_mm256_div_ps(self.0, other.0))
    }
}

impl Sub for F32x8 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(_mm256_sub_ps(self.0, other.0))
    }
}

impl Mul for F32x8 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(_mm256_mul_ps(self.0, other.0))
    }
}

impl MulAssign for F32x8 {
    fn mul_assign(&mut self, other: Self) {
        self.0 = _mm256_mul_ps(self.0, other.0)
    }
}

//impl PartialEq for F32x8 {
//    fn eq(&self, other: &Self) -> bool {
//
//    }
//}

#[derive(Debug, Copy, Clone)]
struct LaneV3 {
    x: LaneF32,
    y: LaneF32,
    z: LaneF32,
}

impl LaneV3 {
    fn new(x: LaneF32, y: LaneF32, z: LaneF32) -> LaneV3 {
        LaneV3 { x, y, z }
    }

    fn new_bcast(x: f32, y: f32, z: f32) -> LaneV3 {
        LaneV3 {
            x: F32x8::from(x),
            y: F32x8::from(y),
            z: F32x8::from(z),
        }
    }

    fn dot(self, other: LaneV3) -> LaneF32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: LaneV3) -> LaneV3 {
        LaneV3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn normalize(self) -> LaneV3 {
        self * (F32x8::from(1.0) / self.len())
    }

    fn reflect(self, normal: LaneV3) -> LaneV3 {
        self - normal * self.dot(normal) * F32x8::from(2.0)
    }

    fn len(self) -> LaneF32 {
        self.dot(self).sqrt()
    }

    // TODO eli: lanebool
    fn is_unit_vector(self) -> bool {
        (self.dot(self) - F32x8::from(1.0)).abs() < TOLERANCE
    }
}

impl Add for LaneV3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Add<LaneF32> for LaneV3 {
    type Output = Self;

    fn add(self, rhs: LaneF32) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl AddAssign for LaneV3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Div<LaneF32> for LaneV3 {
    type Output = Self;

    fn div(self, rhs: LaneF32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Sub for LaneV3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Sub<LaneF32> for LaneV3 {
    type Output = Self;

    fn sub(self, rhs: LaneF32) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Mul for LaneV3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Mul<LaneF32> for LaneV3 {
    type Output = Self;

    fn mul(self, rhs: LaneF32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl MulAssign<LaneF32> for LaneV3 {
    fn mul_assign(&mut self, rhs: LaneF32) {
        *self = Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl MulAssign for LaneV3 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

#[derive(Debug)]
struct Camera {
    origin: LaneV3,
    x: LaneV3,
    y: LaneV3,
    z: LaneV3,
    film_lower_left: LaneV3,
    film_width: LaneF32,
    film_height: LaneF32,
}

impl Camera {
    fn new(look_from: LaneV3, look_at: LaneV3, aspect_ratio: f32) -> Camera {
        assert!(aspect_ratio > 1.0, "width must be greater than height");

        let origin = look_from - look_at;
        let z = origin.normalize();
        let x = LaneV3::new_bcast(0.0, 0.0, 1.0).cross(z).normalize();
        let y = z.cross(x).normalize();

        let film_height = F32x8::from(1.0);
        let film_width = film_height * F32x8::from(aspect_ratio);
        let film_lower_left =
            origin - z - y * F32x8::from(0.5) * film_height - x * F32x8::from(0.5) * film_width;

        Camera {
            origin,
            x,
            y,
            z,
            film_lower_left,
            film_width,
            film_height,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum MaterialType {
    Diffuse,
    Specular,
}

#[derive(Debug, Clone)]
struct Material {
    emit_color: V3,
    reflect_color: V3,
    t: MaterialType,
}

struct Sphere {
    p: V3,
    rsqrd: f32,
    m: Material,
}

impl Sphere {
    fn new(p: V3, r: f32, m: Material) -> Sphere {
        Sphere { p, rsqrd: r * r, m }
    }
}

// https://entropymine.com/imageworsener/srgbformula/
fn linear_to_srgb(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else if x > 0.0031308 {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    } else {
        x * 12.92
    }
}

// pcg xsh rs 64/32 (mcg)
fn pcg(state: &mut u64) -> u32 {
    let s = *state;
    *state = s.wrapping_mul(6364136223846793005);
    (((s >> 22) ^ s) >> ((s >> 61) + 22)) as u32
}

// TODO eli: switch back to xorshift and compute a different rng per lane
fn randf() -> LaneF32 {
    THREAD_RNG.with(|rng_cell| {
        let mut state = rng_cell.get();
        let randu = (pcg(&mut state) >> 9) | 0x3f800000;
        let randf = f32::from_bits(randu) - 1.0;
        rng_cell.set(state);
        F32x8::from(randf)
    })
}

fn randf_range(min: f32, max: f32) -> LaneF32 {
    F32x8::from(min) + F32x8::from(max - min) * randf()
}

fn intersect_world(spheres: &Vec<Sphere>, origin: LaneV3, dir: LaneV3) -> Option<(f32, &Sphere)> {
    let mut hit = None;
    let mut hit_dist = f32::MAX;

    for s in spheres {
        let sphere_relative_origin = origin - s.p;
        let b = dir.dot(sphere_relative_origin);
        let c = sphere_relative_origin.dot(sphere_relative_origin) - s.rsqrd;
        let discr = b * b - c;

        // at least one real root, meaning we've hit the sphere
        if discr > 0.0 {
            let root_term = discr.sqrt();
            // Order here matters. root_term is positive; b may be positive or negative
            //
            // If b is negative, -b is positive, so -b + root_term is _more_ positive than -b - root_term
            // Thus we check -b - root_term first; if it's negative, we check -b + root_term. This is why -b - root_term
            // must be first.
            //
            // Second case is less interesting
            // If b is positive, -b is negative, so -b - root_term is more negative and we will then check -b + root_term
            let t = -b - root_term; // -b minus positive
            if t > TOLERANCE && t < hit_dist {
                hit_dist = t;
                hit = Some((hit_dist, s));
                continue;
            }
            let t = -b + root_term; // -b plus positive
            if t > TOLERANCE && t < hit_dist {
                hit_dist = t;
                hit = Some((hit_dist, s));
                continue;
            }
        }
    }
    hit
}

fn cast(
    bg: &Material,
    spheres: &Vec<Sphere>,
    mut origin: LaneV3,
    mut dir: LaneV3,
    mut bounces: u32,
) -> LaneV3 {
    let mut color = LaneV3::new_bcast(0.0, 0.0, 0.0);
    let mut reflectance = LaneV3::new_bcast(1.0, 1.0, 1.0);

    loop {
        debug_assert!(dir.is_unit_vector());
        let hit = intersect_world(spheres, origin, dir);
        match (hit, bounces) {
            (None, _) => {
                color += reflectance * bg.emit_color;
                break;
            }
            (Some((_, s)), 0) => {
                color += reflectance * s.m.emit_color;
                break;
            }
            (Some((hit_dist, s)), _) => {
                bounces -= 1;
                color += reflectance * s.m.emit_color;
                reflectance *= s.m.reflect_color;
                let hit_point = origin + dir * hit_dist;
                origin = hit_point;
                dir = match s.m.t {
                    MaterialType::Specular => {
                        let hit_normal = (hit_point - s.p).normalize();
                        dir.reflect(hit_normal)
                    }
                    MaterialType::Diffuse => {
                        let a = randf_range(0.0, 2.0 * PI);
                        let z = randf_range(-1.0, 1.0);
                        let r = (1.0 - z * z).sqrt();
                        LaneV3::new_bcast(r * a.cos(), r * a.sin(), z)
                    }
                };
            }
        }
    }
    color
}

thread_local! {
    static THREAD_RNG: Cell<u64> = {
        let mut buf = [0u8; 8];
        getrandom::getrandom(&mut buf).unwrap();
        Cell::new(u64::from_le_bytes(buf))
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Arguments::from_env();
    let rays_per_pixel = args.opt_value_from_str(["-r", "--rays"])?.unwrap_or(100);
    let rays_per_batch = args
        .opt_value_from_str(["-b", "--batch"])?
        .unwrap_or(rays_per_pixel);
    let bounces = args.opt_value_from_str("--bounces")?.unwrap_or(8);
    let filename = args
        .opt_value_from_str("-o")?
        .unwrap_or("out.png".to_string());
    args.finish()?;

    assert!(
        rays_per_batch <= rays_per_pixel,
        "number of rays in batch cannot exceed total rays"
    );

    // Materials
    let bg = Material {
        emit_color: V3(0.3, 0.4, 0.8),
        reflect_color: V3(0.0, 0.0, 0.0),
        t: MaterialType::Specular,
    };
    let ground = Material {
        emit_color: V3(0.0, 0.0, 0.0),
        reflect_color: V3(0.5, 0.5, 0.5),
        t: MaterialType::Diffuse,
    };
    let left = Material {
        emit_color: V3(0.0, 0.0, 0.0),
        reflect_color: V3(1.0, 0.0, 0.0),
        t: MaterialType::Specular,
    };
    let center = Material {
        emit_color: V3(0.4, 0.8, 0.9),
        reflect_color: V3(0.8, 0.8, 0.8),
        t: MaterialType::Specular,
    };
    let right = Material {
        emit_color: V3(0.0, 0.0, 0.0),
        reflect_color: V3(0.95, 0.95, 0.95),
        t: MaterialType::Specular,
    };

    let spheres = vec![
        Sphere::new(V3(0.0, 0.0, -100.0), 100.0, ground),
        Sphere::new(V3(0.0, 0.0, 1.0), 1.0, center),
        Sphere::new(V3(-2.0, -3.0, 1.5), 0.3, right.clone()),
        Sphere::new(V3(-3.0, -6.0, 0.0), 0.3, right.clone()),
        Sphere::new(V3(-3.0, -5.0, 2.0), 0.5, left),
        Sphere::new(V3(3.0, -3.0, 0.8), 1.0, right),
    ];

    let width = 1920;
    let height = 1080;
    let inv_width = 1.0 / (width as f32 - 1.0);
    let inv_height = 1.0 / (height as f32 - 1.0);
    let mut pixels = vec![V3(0.0, 0.0, 0.0); width * height];
    let cam = Camera::new(
        LaneV3::new_bcast(0.0, -10.0, 1.0),
        LaneV3::new_bcast(0.0, 0.0, 0.0),
        width as f32 / height as f32,
    );

    let batches = if rays_per_pixel % rays_per_batch == 0 {
        rays_per_pixel / rays_per_batch
    } else {
        rays_per_pixel / rays_per_batch + 1
    };

    let start = Instant::now();
    let mut rays_shot = 0;
    for b in 0..batches {
        let batch_size = cmp::min(rays_per_batch, rays_per_pixel - rays_shot);

        // TODO eli: problem here: if the pixels is a group of lanes then our image_x/y math will be off here
        // probably need to keep the pixels array as flat v3s then do the batching inside this loop
        pixels.par_iter_mut().enumerate().for_each(|(i, color)| {
            // need to iter by chunks then pack all the image_x/y into an f32x8
            let image_x = (i % width) as f32;
            let image_y = (height - (i / width) - 1) as f32; // flip image right-side-up
            for _ in 0..batch_size {
                // calculate ratio we've moved along the image (y/height), step proportionally within the film
                let rand_x = randf();
                let rand_y = randf();
                let film_x = cam.x * cam.film_width * (image_x + rand_x) * inv_width;
                let film_y = cam.y * cam.film_height * (image_y + rand_y) * inv_height;
                let film_p = cam.film_lower_left + film_x + film_y;

                // remember that a pixel in float-space is a _range_. We want to send multiple rays within that range
                // to do this we take the start of that range (what we calculated as the image projecting onto our film),
                // then add a random [0,1) float
                let ray_p = cam.origin;
                let ray_dir = (film_p - cam.origin).normalize();
                *color += cast(&bg, &spheres, ray_p, ray_dir, bounces);
            }
        });
        rays_shot += batch_size;
        println!("Shot {} of {} rays", rays_shot, rays_per_pixel);

        let mut buf: Vec<u8> = Vec::with_capacity(width * height * 3);
        for p in &pixels {
            buf.push((255.0 * linear_to_srgb(p.0 / rays_shot as f32)) as u8);
            buf.push((255.0 * linear_to_srgb(p.1 / rays_shot as f32)) as u8);
            buf.push((255.0 * linear_to_srgb(p.2 / rays_shot as f32)) as u8);
        }

        let f = format!("{}-{}", b, filename);
        image::save_buffer(f, &buf, width as u32, height as u32, image::ColorType::Rgb8)?;
    }
    println!("Rendering took {:.3}s", start.elapsed().as_secs_f32());
    Ok(())
}
