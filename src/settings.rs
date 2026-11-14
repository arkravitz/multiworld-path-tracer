use crate::scene::WorldPreset;
use clap::ValueEnum;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[repr(u32)]
pub enum Mode {
    WorldA,
    WorldB,
    Blend,
    PerPath,
    PerBounce,
    Persistent,
}
impl Mode {
    pub const ALL: [Self; 6] = [
        Self::WorldA,
        Self::WorldB,
        Self::Blend,
        Self::PerPath,
        Self::PerBounce,
        Self::Persistent,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::WorldA => "1 · World A only",
            Self::WorldB => "2 · World B only",
            Self::Blend => "3 · Image-space blend",
            Self::PerPath => "4 · Once per path",
            Self::PerBounce => "5 · Independent per bounce",
            Self::Persistent => "6 · Persistent switching",
        }
    }
    pub fn slug(self) -> &'static str {
        match self {
            Self::WorldA => "01-world-a",
            Self::WorldB => "02-world-b",
            Self::Blend => "03-blend",
            Self::PerPath => "04-per-path",
            Self::PerBounce => "05-per-bounce",
            Self::Persistent => "06-persistent",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum View {
    Radiance,
    PrimaryWorld,
    Switches,
    Bounces,
    Normals,
}
impl View {
    pub const ALL: [Self; 5] = [
        Self::Radiance,
        Self::PrimaryWorld,
        Self::Switches,
        Self::Bounces,
        Self::Normals,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Radiance => "Radiance",
            Self::PrimaryWorld => "Primary world",
            Self::Switches => "World switches",
            Self::Bounces => "Bounce count",
            Self::Normals => "Surface normals",
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub mode: Mode,
    pub worlds: [WorldPreset; 2],
    pub use_bvh: bool,
    pub light_sampling: bool,
    pub adaptive_bvh: bool,
    pub denoise: bool,
    pub denoise_strength: f32,
    pub p: f32,
    pub persistence: f32,
    pub spp: u32,
    pub depth: u32,
    pub exposure: f32,
    pub seed: u32,
    pub view: View,
    pub spatial: bool,
    pub primary_a: bool,
    pub threshold: f32,
    pub frequency: f32,
    pub softness: f32,
    pub time: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            mode: Mode::PerBounce,
            worlds: [WorldPreset::WarmSpheres, WorldPreset::CoolBoxes],
            use_bvh: true,
            light_sampling: true,
            adaptive_bvh: true,
            denoise: false,
            denoise_strength: 0.8,
            p: 0.5,
            persistence: 0.8,
            spp: 1,
            depth: 10,
            exposure: 0.0,
            seed: 42,
            view: View::Radiance,
            spatial: false,
            primary_a: false,
            threshold: 0.5,
            frequency: 0.35,
            softness: 0.2,
            time: 0.0,
            yaw: 0.62,
            pitch: 0.38,
            distance: 18.0,
        }
    }
}
impl Settings {
    /// Display changes, batch size, and blend weight do not invalidate linear sums.
    pub fn invalidates(&self, old: &Self) -> bool {
        let mut a = self.clone();
        let mut b = old.clone();
        a.denoise = false;
        b.denoise = false;
        a.denoise_strength = 0.;
        b.denoise_strength = 0.;
        a.exposure = 0.;
        b.exposure = 0.;
        a.spp = 0;
        b.spp = 0;
        // Debug statistics are always accumulated independently of the displayed view.
        a.view = View::Radiance;
        b.view = View::Radiance;
        if a.mode == Mode::Blend && b.mode == Mode::Blend {
            a.p = 0.;
            b.p = 0.;
        }
        a != b
    }
    /// Camera motion may reproject history. Transport/scene/mixture changes may
    /// not; even blend p changes invalidate filtered composite history.
    pub fn history_compatible(&self, old: &Self) -> bool {
        let mut a = self.clone();
        let mut b = old.clone();
        a.yaw = 0.;
        b.yaw = 0.;
        a.pitch = 0.;
        b.pitch = 0.;
        a.distance = 0.;
        b.distance = 0.;
        a.spp = 0;
        b.spp = 0;
        a.exposure = 0.;
        b.exposure = 0.;
        a.view = View::Radiance;
        b.view = View::Radiance;
        a == b
    }
    /// Horizontal orbit is cyclic; a control's numeric endpoint must never block
    /// more rotation. Keep angles small for stable trigonometry over many turns.
    pub fn set_orbit_yaw(&mut self, yaw: f32) {
        if yaw.is_finite() {
            self.yaw = (yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
        }
    }
    pub fn orbit(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.set_orbit_yaw(self.yaw + yaw_delta);
        self.pitch = (self.pitch + pitch_delta).clamp(0.05, 1.4);
    }
    pub fn camera(&self, aspect: f32) -> [[f32; 4]; 4] {
        let target = Vec3::new(0., 1., 0.);
        let eye = target
            + self.distance
                * Vec3::new(
                    self.pitch.cos() * self.yaw.sin(),
                    self.pitch.sin(),
                    self.pitch.cos() * self.yaw.cos(),
                );
        let forward = (target - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize() * (40_f32.to_radians() / 2.).tan() * aspect;
        let up = right.normalize().cross(forward) * (40_f32.to_radians() / 2.).tan();
        [
            eye.extend(0.).to_array(),
            forward.extend(0.).to_array(),
            right.extend(0.).to_array(),
            up.extend(0.).to_array(),
        ]
    }
}
