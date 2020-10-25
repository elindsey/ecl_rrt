//#![feature(link_llvm_intrinsics)]

use pico_args::Arguments;
use rayon::prelude::*;
use std::{
    cell::Cell,
    cmp,
    f32::consts::PI,
    ops::{Add, AddAssign, BitAnd, BitOr, BitXorAssign, Div, Mul, MulAssign, Neg, Shl, Shr, Sub},
    time::Instant,
};

const TOLERANCE: f32 = 0.0001;
const WIDTH: usize = 8;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

//extern "C" {
//    #[link_name = "llvm.cos.v8f32"]
//    fn cos(a: WideF32) -> WideF32;
//}

//#[cfg(all(
//    any(target_arch = "x86", target_arch = "x86_64"),
//    target_feature = "avx2"
//))]
#[derive(Debug, Copy, Clone)]
struct V3(f32, f32, f32);

#[derive(Debug, Copy, Clone)]
struct WideF32(__m256);

impl WideF32 {
    //fn abs(&self) -> Self {
    //}

    fn select(x: WideF32, y: WideF32, mask: WideF32) -> WideF32 {
        WideF32(_mm256_blendv_ps(x.0, y.0, mask.0))
    }

    fn cos(&self) -> Self {
        // TODO eli: implement this
        //unsafe { cos(self) }
        self.clone()
    }

    fn sin(&self) -> Self {
        // TODO eli: implement this
        self.clone()
    }

    fn sqrt(&self) -> Self {
        Self(_mm256_sqrt_ps(self.0))
    }

    fn gt(&self, other: Self) -> Self {
        Self(_mm256_cmp_ps(self.0, other.0, _CMP_GT_OQ))
    }

    fn lt(&self, other: Self) -> Self {
        Self(_mm256_cmp_ps(self.0, other.0, _CMP_LT_OQ))
    }
}

impl From<f32> for WideF32 {
    fn from(x: f32) -> Self {
        Self(_mm256_set1_ps(x))
    }
}

impl From<[f32; 8]> for WideF32 {
    fn from(x: [f32; 8]) -> Self {
        Self(_mm256_load_ps(x.as_ptr()))
    }
}

impl Add for WideF32 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(_mm256_add_ps(self.0, other.0))
    }
}

impl AddAssign for WideF32 {
    fn add_assign(&mut self, other: Self) {
        self.0 = _mm256_add_ps(self.0, other.0)
    }
}

impl BitAnd for WideF32 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(_mm256_and_ps(self.0, other.0))
    }
}

impl BitOr for WideF32 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(_mm256_or_ps(self.0, other.0))
    }
}

impl Div for WideF32 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(_mm256_div_ps(self.0, other.0))
    }
}

impl Sub for WideF32 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(_mm256_sub_ps(self.0, other.0))
    }
}

impl Mul for WideF32 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(_mm256_mul_ps(self.0, other.0))
    }
}

impl MulAssign for WideF32 {
    fn mul_assign(&mut self, other: Self) {
        self.0 = _mm256_mul_ps(self.0, other.0)
    }
}

impl Neg for WideF32 {
    type Output = Self;

    fn neg(self) -> Self {
        Self(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)))
    }
}

#[derive(Debug, Copy, Clone)]
struct WideU32(__m256i);

impl From<i32> for WideU32 {
    fn from(x: i32) -> Self {
        Self(_mm256_set1_epi32(x))
    }
}

impl From<&[u8; 32]> for WideU32 {
    fn from(x: &[u8; 32]) -> Self {
        // TODO eli: I should probably just use bytemuck
        Self(unsafe { std::mem::transmute(x.as_ptr()) })
    }
}

impl Shl<i32> for WideU32 {
    type Output = Self;

    fn shl(self, shift: i32) -> Self {
        Self(_mm256_slli_epi32(self.0, shift))
    }
}

impl Shr<i32> for WideU32 {
    type Output = Self;

    fn shr(self, shift: i32) -> Self {
        Self(_mm256_srli_epi32(self.0, shift))
    }
}

impl BitOr for WideU32 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(_mm256_or_si256(self.0, other.0))
    }
}

impl BitXorAssign for WideU32 {
    fn bitxor_assign(&mut self, other: Self) {
        self.0 = _mm256_xor_si256(self.0, other.0)
    }
}

#[derive(Debug, Copy, Clone)]
struct WideV3 {
    x: WideF32,
    y: WideF32,
    z: WideF32,
}

impl WideV3 {
    fn new(x: WideF32, y: WideF32, z: WideF32) -> WideV3 {
        WideV3 { x, y, z }
    }

    fn new_bcast(x: f32, y: f32, z: f32) -> WideV3 {
        WideV3 {
            x: WideF32::from(x),
            y: WideF32::from(y),
            z: WideF32::from(z),
        }
    }

    fn dot(self, other: WideV3) -> WideF32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: WideV3) -> WideV3 {
        WideV3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn normalize(self) -> WideV3 {
        self * (WideF32::from(1.0) / self.len())
    }

    fn reflect(self, normal: WideV3) -> WideV3 {
        self - normal * self.dot(normal) * WideF32::from(2.0)
    }

    fn len(self) -> WideF32 {
        self.dot(self).sqrt()
    }

    // TODO eli: lanebool
    //fn is_unit_vector(self) -> bool {
    //    (self.dot(self) - WideF32::from(1.0)).abs() < TOLERANCE
    //}
}

impl Add for WideV3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Add<WideF32> for WideV3 {
    type Output = Self;

    fn add(self, rhs: WideF32) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl AddAssign for WideV3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Div<WideF32> for WideV3 {
    type Output = Self;

    fn div(self, rhs: WideF32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Sub for WideV3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Sub<WideF32> for WideV3 {
    type Output = Self;

    fn sub(self, rhs: WideF32) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl Mul for WideV3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Mul<WideF32> for WideV3 {
    type Output = Self;

    fn mul(self, rhs: WideF32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl MulAssign<WideF32> for WideV3 {
    fn mul_assign(&mut self, rhs: WideF32) {
        *self = Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl MulAssign for WideV3 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

#[derive(Debug)]
struct Camera {
    origin: WideV3,
    x: WideV3,
    y: WideV3,
    z: WideV3,
    film_lower_left: WideV3,
    film_width: WideF32,
    film_height: WideF32,
}

impl Camera {
    fn new(look_from: WideV3, look_at: WideV3, aspect_ratio: f32) -> Camera {
        assert!(aspect_ratio > 1.0, "width must be greater than height");

        let origin = look_from - look_at;
        let z = origin.normalize();
        let x = WideV3::new_bcast(0.0, 0.0, 1.0).cross(z).normalize();
        let y = z.cross(x).normalize();

        let film_height = WideF32::from(1.0);
        let film_width = film_height * WideF32::from(aspect_ratio);
        let film_lower_left =
            origin - z - y * WideF32::from(0.5) * film_height - x * WideF32::from(0.5) * film_width;

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
    p: WideV3,
    rsqrd: WideF32,
    m: Material,
}

impl Sphere {
    fn new(p: V3, r: f32, m: Material) -> Sphere {
        Sphere {
            p: WideV3::new_bcast(p.0, p.1, p.2),
            rsqrd: WideF32::from(r * r),
            m,
        }
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

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
fn xorshift(state: &mut WideU32) -> WideU32 {
    //debug_assert!(*state != 0, "xorshift cannot be seeded with 0");
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

// pcg xsh rs 64/32 (mcg)
fn pcg(state: &mut u64) -> u32 {
    let s = *state;
    *state = s.wrapping_mul(6364136223846793005);
    (((s >> 22) ^ s) >> ((s >> 61) + 22)) as u32
}

// TODO eli: switch back to xorshift and compute a different rng per lane
fn randf() -> WideF32 {
    THREAD_RNG.with(|rng_cell| {
        let mut state = rng_cell.get();
        let randu = (xorshift(&mut state) >> 9) | WideU32::from(0x3f800000);
        let randf = WideF32(_mm256_castsi256_ps(randu.0)) - WideF32::from(1.0);
        rng_cell.set(state);
        WideF32::from(randf)
    })
}

fn randf_range(min: f32, max: f32) -> WideF32 {
    WideF32::from(min) + WideF32::from(max - min) * randf()
}

fn raycast(
    bg: &Material,
    spheres: &Vec<Sphere>,
    mut origin: WideV3,
    mut dir: WideV3,
    mut bounces: u32,
) -> WideV3 {
    let mut color = WideV3::new_bcast(0.0, 0.0, 0.0);
    let mut reflectance = WideV3::new_bcast(1.0, 1.0, 1.0);

    loop {
        //debug_assert!(dir.is_unit_vector());
        let mut hit = WideU32::from(0);
        let mut hit_dist = WideF32::from(f32::MAX);

        for s in spheres {
            let sphere_relative_origin = origin - s.p;
            let b = dir.dot(sphere_relative_origin);
            let c = sphere_relative_origin.dot(sphere_relative_origin) - s.rsqrd;
            let discr = b * b - c;

            // at least one real root, meaning we've hit the sphere
            let discrmask = discr.gt(WideF32::from(0.0));
            // TODO eli: short circuit if mask is all empty
            let root_term = discr.sqrt();
            // Order here matters, must consider t0 first. root_term is positive; b may be positive or negative
            //
            // If b is negative, -b is positive, so -b + root_term is _more_ positive than -b - root_term
            // Thus we check -b - root_term first; if it's negative, we check -b + root_term. This is why -b - root_term
            // must be first.
            //
            // Second case is less interesting
            // If b is positive, -b is negative, so -b - root_term is more negative and we will then check -b + root_term

            // TODO eli: move -b redundancy out
            let t0 = -b - root_term;
            let t1 = -b + root_term;

            // t0 if hit, else t1
            let t = WideF32::select(t1, t0, t0.gt(WideF32::from(TOLERANCE)));
            // TODO eli: these masks might be better as wideu32
            let mask = discrmask & t.gt(WideF32::from(TOLERANCE)) & t.lt(hit_dist);
            // TODO eli: '??' needs to be some way to identify the sphere, probably need to change that access pattern
            hit = select(hit, ??, mask);
            hit_dist = WideF32::select(hit_dist, t, mask);

            // TODO eli: conditional assign
            // hit_dist = t;
            // hit = Some((hit_dist, s));
        }

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
                        WideV3::new_bcast(r * a.cos(), r * a.sin(), z)
                    }
                };
            }
        }
    }
    color
}

thread_local! {
    // TODO eli: this probably needs to be a refcell now
    static THREAD_RNG: Cell<WideU32> = {
        let mut buf = [0u8; 32];
        getrandom::getrandom(&mut buf).unwrap();
        Cell::new(WideU32::from(&buf))
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
    // TODO eli: bugs when pixels % WIDTH != 0
    let mut pixels = vec![WideV3::new_bcast(0.0, 0.0, 0.0); width * height / WIDTH];
    let cam = Camera::new(
        WideV3::new_bcast(0.0, -10.0, 1.0),
        WideV3::new_bcast(0.0, 0.0, 0.0),
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
        pixels.par_iter_mut().enumerate().for_each(|(i, colors)| {
            // need to iter by chunks then pack all the image_x/y into an f32x8
            let mut i = i * WIDTH;
            let xs: [f32; WIDTH];
            let ys: [f32; WIDTH];
            // TODO eli: this is terrible
            for idx in 0..WIDTH {
                xs[idx] = (i % width) as f32;
                ys[idx] = (height - (i / width) - 1) as f32;
                i += 1;
            }
            let image_x = WideF32::from(xs);
            let image_y = WideF32::from(ys);
            //let image_x = (i % width) as f32;
            //let image_y = (height - (i / width) - 1) as f32; // flip image right-side-up
            for _ in 0..batch_size {
                // calculate ratio we've moved along the image (y/height), step proportionally within the film
                let rand_x = randf();
                let rand_y = randf();
                let film_x = cam.x * cam.film_width * (image_x + rand_x) * WideF32::from(inv_width);
                let film_y =
                    cam.y * cam.film_height * (image_y + rand_y) * WideF32::from(inv_height);
                let film_p = cam.film_lower_left + film_x + film_y;

                // remember that a pixel in float-space is a _range_. We want to send multiple rays within that range
                // to do this we take the start of that range (what we calculated as the image projecting onto our film),
                // then add a random [0,1) float
                let ray_p = cam.origin;
                let ray_dir = (film_p - cam.origin).normalize();
                *colors += raycast(&bg, &spheres, ray_p, ray_dir, bounces);
            }
        });
        rays_shot += batch_size;
        println!("Shot {} of {} rays", rays_shot, rays_per_pixel);

        // TODO eli: it's easier to do a pixel at a time, simd the rays, then horizontal add them to get a color
        // that would require reworking some of the incremental batching though
        let mut buf: Vec<u8> = Vec::with_capacity(width * height * 3);
        for p in &pixels {
            // maybe? https://users.rust-lang.org/t/correct-way-to-use-the-simd-intrinsics/42271/9
            // hard to tell what is and isn't UB in rust
            let x: [f32; 8] = unsafe { std::mem::transmute(p.x) };
            let y: [f32; 8] = unsafe { std::mem::transmute(p.y) };
            let z: [f32; 8] = unsafe { std::mem::transmute(p.z) };
            for idx in 0..8 {
                buf.push((255.0 * linear_to_srgb(x[idx] / rays_shot as f32)) as u8);
                buf.push((255.0 * linear_to_srgb(y[idx] / rays_shot as f32)) as u8);
                buf.push((255.0 * linear_to_srgb(z[idx] / rays_shot as f32)) as u8);
            }
        }

        let f = format!("{}-{}", b, filename);
        image::save_buffer(f, &buf, width as u32, height as u32, image::ColorType::Rgb8)?;
    }
    println!("Rendering took {:.3}s", start.elapsed().as_secs_f32());
    Ok(())
}
