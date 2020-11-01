use pico_args::Arguments;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{
    cell::Cell,
    cmp,
    f32::consts::PI,
    ops::Neg,
    ops::{Add, AddAssign, BitAnd, BitOr, Div, Mul, MulAssign, Sub},
    time::Instant,
};

const TOLERANCE: f32 = 0.0001;
const SIMD_WIDTH: usize = 8;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

struct Spheres {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    rsqrds: Vec<f32>,
    mats: Vec<Material>,
}

impl Spheres {
    fn new(spheres: Vec<Sphere>) -> Self {
        let len = (spheres.len() + SIMD_WIDTH - 1) / SIMD_WIDTH * SIMD_WIDTH;

        let mut me = Self {
            xs: Vec::with_capacity(len),
            ys: Vec::with_capacity(len),
            zs: Vec::with_capacity(len),
            rsqrds: Vec::with_capacity(len),
            mats: Vec::with_capacity(len),
        };

        for s in spheres {
            me.xs.push(s.p.0);
            me.ys.push(s.p.1);
            me.zs.push(s.p.2);
            me.rsqrds.push(s.rsqrd);
            me.mats.push(s.m);
        }

        // pad everything out to the simd width
        me.xs.resize(len, 0.0);
        me.ys.resize(len, 0.0);
        me.zs.resize(len, 0.0);
        me.rsqrds.resize(len, 0.0);

        let default_mat = Material {
            emit_color: V3(0.0, 0.0, 0.0),
            reflect_color: V3(0.0, 0.0, 0.0),
            t: MaterialType::Specular,
        };
        me.mats.resize(len, default_mat);

        me
    }

    fn len(&self) -> usize {
        self.xs.len()
    }
}

#[derive(Debug, Copy, Clone)]
struct WideI32(__m256i);

impl WideI32 {
    fn new(e7: i32, e6: i32, e5: i32, e4: i32, e3: i32, e2: i32, e1: i32, e0: i32) -> Self {
        Self(unsafe { _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0) })
    }

    fn splat(x: i32) -> Self {
        Self(unsafe { _mm256_set1_epi32(x) })
    }

    fn select(x: WideI32, y: WideI32, mask: WideF32) -> WideI32 {
        WideI32(unsafe {
            _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(x.0),
                _mm256_castsi256_ps(y.0),
                mask.0,
            ))
        })
    }
}

impl Add for WideI32 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(unsafe { _mm256_add_epi32(self.0, other.0) })
    }
}

impl AddAssign for WideI32 {
    fn add_assign(&mut self, other: Self) {
        self.0 = unsafe { _mm256_add_epi32(self.0, other.0) }
    }
}

#[derive(Debug, Copy, Clone)]
struct WideF32(__m256);

impl WideF32 {
    fn load(x: &[f32]) -> Self {
        debug_assert!(x.len() >= SIMD_WIDTH);
        // aligning this would require some hoops on vec alloc
        // https://stackoverflow.com/questions/60180121/how-do-i-allocate-a-vecu8-that-is-aligned-to-the-size-of-the-cache-line
        Self(unsafe { _mm256_loadu_ps(x.as_ptr()) })
    }

    fn any(&self) -> bool {
        (unsafe { _mm256_movemask_ps(self.0) }) != 0
    }

    fn hmin(&self) -> f32 {
        // TODO(eli): tests
        unsafe {
            /*
            This can be done entirely in avx with permute2f128, but that is allegedly very
            slow on AMD prior to Zen2 (and is anecdotally slower on my Intels as well)

            initial m256
            1 2 3 4 5 6 7 8

            extract half, cast the other half down to m128, min
              1 2 3 4
              5 6 7 8
            = 1 2 3 4

            permute backwards, min
              1 2 3 4
              4 3 2 1
            = 1 2 2 1

            unpack hi, min
              1 2 2 1
              1 1 2 2
            = 1 1 2 1
            */
            let x = self.0;
            let y = _mm256_extractf128_ps(x, 1);
            let m1 = _mm_min_ps(_mm256_castps256_ps128(x), y);
            let m2 = _mm_permute_ps(m1, 27);
            let m2 = _mm_min_ps(m1, m2);
            let m3 = _mm_unpackhi_ps(m2, m2);
            let m = _mm_min_ps(m2, m3);
            _mm_cvtss_f32(m)
        }
    }

    fn splat(x: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(x) })
    }

    fn select(x: WideF32, y: WideF32, mask: WideF32) -> WideF32 {
        WideF32(unsafe { _mm256_blendv_ps(x.0, y.0, mask.0) })
    }

    fn sqrt(&self) -> Self {
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }

    fn gt(&self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, other.0, _CMP_GT_OQ) })
    }

    fn lt(&self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, other.0, _CMP_LT_OQ) })
    }

    fn eq(&self, other: Self) -> Self {
        Self(unsafe { _mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ) })
    }
}

impl Add for WideF32 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(unsafe { _mm256_add_ps(self.0, other.0) })
    }
}

impl AddAssign for WideF32 {
    fn add_assign(&mut self, other: Self) {
        self.0 = unsafe { _mm256_add_ps(self.0, other.0) }
    }
}

impl BitAnd for WideF32 {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(unsafe { _mm256_and_ps(self.0, other.0) })
    }
}

impl BitOr for WideF32 {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(unsafe { _mm256_or_ps(self.0, other.0) })
    }
}

impl Div for WideF32 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(unsafe { _mm256_div_ps(self.0, other.0) })
    }
}

impl Sub for WideF32 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(unsafe { _mm256_sub_ps(self.0, other.0) })
    }
}

impl Mul for WideF32 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(unsafe { _mm256_mul_ps(self.0, other.0) })
    }
}

impl MulAssign for WideF32 {
    fn mul_assign(&mut self, other: Self) {
        self.0 = unsafe { _mm256_mul_ps(self.0, other.0) }
    }
}

impl Neg for WideF32 {
    type Output = Self;

    fn neg(self) -> Self {
        Self(unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) })
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
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
        (self.dot(self) - 1.0).abs() < TOLERANCE
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

    fn add(self, rhs: f32) -> Self {
        Self(self.0 + rhs, self.1 + rhs, self.2 + rhs)
    }
}

impl AddAssign for V3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl Div<f32> for V3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self(self.0 / rhs, self.1 / rhs, self.2 / rhs)
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

    fn sub(self, rhs: f32) -> Self {
        Self(self.0 - rhs, self.1 - rhs, self.2 - rhs)
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

    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl MulAssign<f32> for V3 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl MulAssign for V3 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

#[derive(Debug)]
struct Camera {
    origin: V3,
    x: V3,
    y: V3,
    z: V3,
    film_lower_left: V3,
    film_width: f32,
    film_height: f32,
}

impl Camera {
    fn new(look_from: V3, look_at: V3, aspect_ratio: f32) -> Camera {
        assert!(aspect_ratio > 1.0, "width must be greater than height");

        let origin = look_from - look_at;
        let z = origin.normalize();
        let x = V3(0.0, 0.0, 1.0).cross(z).normalize();
        let y = z.cross(x).normalize();

        let film_height = 1.0;
        let film_width = film_height * aspect_ratio;
        let film_lower_left = origin - z - y * 0.5 * film_height - x * 0.5 * film_width;

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

#[derive(Debug, Clone, PartialEq)]
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

fn randf() -> f32 {
    // TODO(eli): thread local perf is terrible; causes function call and branching
    THREAD_RNG.with(|rng_cell| {
        let mut state = rng_cell.get();
        let randu = (pcg(&mut state) >> 9) | 0x3f800000;
        let randf = f32::from_bits(randu) - 1.0;
        rng_cell.set(state);
        randf
    })
}

fn randf_range(min: f32, max: f32) -> f32 {
    min + (max - min) * randf()
}

fn cast(bg: &Material, spheres: &Spheres, mut origin: V3, mut dir: V3, mut bounces: u32) -> V3 {
    let mut color = V3(0.0, 0.0, 0.0);
    let mut reflectance = V3(1.0, 1.0, 1.0);

    loop {
        debug_assert!(dir.is_unit_vector());
        let ox = WideF32::splat(origin.0);
        let oy = WideF32::splat(origin.1);
        let oz = WideF32::splat(origin.2);
        let mut hit = None;
        let mut hits = WideI32::splat(-1);
        let mut hit_dists = WideF32::splat(f32::MAX);

        let dirx = WideF32::splat(dir.0);
        let diry = WideF32::splat(dir.1);
        let dirz = WideF32::splat(dir.2);
        // TODO(eli): should be a wideu32
        let mut wide_ids = WideI32::new(7, 6, 5, 4, 3, 2, 1, 0);

        // TODO(eli): egregious bounds checking here
        for i in (0..spheres.len()).step_by(SIMD_WIDTH) {
            let wide_xs = WideF32::load(&spheres.xs[i..i + SIMD_WIDTH]);
            let wide_ys = WideF32::load(&spheres.ys[i..i + SIMD_WIDTH]);
            let wide_zs = WideF32::load(&spheres.zs[i..i + SIMD_WIDTH]);
            let wide_rsqrds = WideF32::load(&spheres.rsqrds[i..i + SIMD_WIDTH]);

            let sphere_relative_x = wide_xs - ox;
            let sphere_relative_y = wide_ys - oy;
            let sphere_relative_z = wide_zs - oz;
            let neg_b =
                dirx * sphere_relative_x + diry * sphere_relative_y + dirz * sphere_relative_z;
            let c = sphere_relative_x * sphere_relative_x
                + sphere_relative_y * sphere_relative_y
                + sphere_relative_z * sphere_relative_z
                - wide_rsqrds;
            let discr = neg_b * neg_b - c;

            let discrmask = discr.gt(WideF32::splat(0.0));
            if discrmask.any() {
                let root_term = discr.sqrt();
                let t0 = neg_b - root_term;
                let t1 = neg_b + root_term;

                // t0 if hit, else t1
                let t = WideF32::select(t1, t0, t0.gt(WideF32::splat(TOLERANCE)));
                let mask = discrmask & t.gt(WideF32::splat(TOLERANCE)) & t.lt(hit_dists);
                hits = WideI32::select(hits, wide_ids, mask);
                hit_dists = WideF32::select(hit_dists, t, mask);
            }
            wide_ids += WideI32::splat(SIMD_WIDTH as i32);
        }
        let hmin = hit_dists.hmin();
        if hmin < f32::MAX {
            let minmask = hit_dists.eq(WideF32::splat(hmin));
            let m = unsafe { _mm256_movemask_ps(minmask.0) };
            let min_idx = m.trailing_zeros() as usize;

            let hit_ids_arr: [i32; 8] = unsafe { std::mem::transmute(hits.0) };
            let hit_dists_arr: [f32; 8] = unsafe { std::mem::transmute(hit_dists.0) };
            hit = Some((hit_dists_arr[min_idx], hit_ids_arr[min_idx] as usize))
        }

        match (hit, bounces) {
            (None, _) => {
                color += reflectance * bg.emit_color;
                break;
            }
            (Some((_, id)), 0) => {
                color += reflectance * spheres.mats[id].emit_color;
                break;
            }
            (Some((hit_dist, id)), _) => {
                bounces -= 1;
                let mat = &spheres.mats[id];
                color += reflectance * mat.emit_color;
                reflectance *= mat.reflect_color;
                let hit_point = origin + dir * hit_dist;
                origin = hit_point;
                dir = match mat.t {
                    MaterialType::Specular => {
                        let sp = V3(spheres.xs[id], spheres.ys[id], spheres.zs[id]);
                        let hit_normal = (hit_point - sp).normalize();
                        dir.reflect(hit_normal)
                    }
                    MaterialType::Diffuse => {
                        let a = randf_range(0.0, 2.0 * PI);
                        let z = randf_range(-1.0, 1.0);
                        let r = (1.0 - z * z).sqrt();
                        V3(r * a.cos(), r * a.sin(), z)
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
    // flush denormals to zero
    unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) };

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

    let spheres = Spheres::new(vec![
        Sphere::new(V3(0.0, 0.0, -100.0), 100.0, ground),
        Sphere::new(V3(0.0, 0.0, 1.0), 1.0, center),
        Sphere::new(V3(-2.0, -3.0, 1.5), 0.3, right.clone()),
        Sphere::new(V3(-3.0, -6.0, 0.0), 0.3, right.clone()),
        Sphere::new(V3(-3.0, -5.0, 2.0), 0.5, left.clone()),
        Sphere::new(V3(3.0, -3.0, 0.8), 1.0, right.clone()),
        Sphere::new(V3(-3.0, -3.0, 2.0), 0.5, left),
        //Sphere::new(V3(5.0, -3.0, 0.8), 1.0, right),
    ]);

    let width = 1920;
    let height = 1080;
    let inv_width = 1.0 / (width as f32 - 1.0);
    let inv_height = 1.0 / (height as f32 - 1.0);
    let mut pixels = vec![V3(0.0, 0.0, 0.0); width * height];
    let cam = Camera::new(
        V3(0.0, -10.0, 1.0),
        V3(0.0, 0.0, 0.0),
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

        pixels.par_iter_mut().enumerate().for_each(|(i, color)| {
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
            let c = *p / rays_shot as f32;
            buf.push((255.0 * linear_to_srgb(c.0)) as u8);
            buf.push((255.0 * linear_to_srgb(c.1)) as u8);
            buf.push((255.0 * linear_to_srgb(c.2)) as u8);
        }

        let f = format!("{}-{}", b, filename);
        image::save_buffer(f, &buf, width as u32, height as u32, image::ColorType::Rgb8)?;
    }
    println!("Rendering took {:.3}s", start.elapsed().as_secs_f32());
    Ok(())
}
