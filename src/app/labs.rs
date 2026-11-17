use crate::{
    scene::{GlassPreset, WorldPreset},
    settings::{DenoiserKind, Mode, SamplingMode, Settings, View},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lab {
    Multiworld,
    Glass,
}

/// Each workspace owns its scene, camera and rendering choices. The engine and
/// image targets remain shared, and switching restarts their accumulation.
pub(super) struct Labs {
    pub active: Lab,
    other: Settings,
}

pub(super) fn glass_settings(seed: u32) -> Settings {
    Settings {
        mode: Mode::WorldA,
        worlds: [WorldPreset::GlassGallery, WorldPreset::CrystalGarden],
        glass: [GlassPreset::Assorted; 2],
        caustics: true,
        photon_guiding: true,
        denoise: true,
        denoiser: DenoiserKind::Oidn,
        // Uniform OIDN wins our short equal-time glass comparison. Adaptive
        // modes remain selectable rather than charging every preview for them.
        sampling: SamplingMode::Uniform,
        depth: 20,
        yaw: 0.6,
        pitch: 0.55,
        distance: 13.,
        exposure: 0.25,
        seed,
        ..Default::default()
    }
}

impl Labs {
    pub fn new(settings: &mut Settings) -> Self {
        let glass = (settings.caustics || settings.photon_nrc)
            && matches!(settings.mode, Mode::WorldA | Mode::WorldB | Mode::Blend);
        if glass {
            settings.neural_cache = false;
            settings.spatial = false;
            settings.primary_a = false;
            Self {
                active: Lab::Glass,
                other: Settings {
                    worlds: [WorldPreset::SunkenTemple, WorldPreset::BioluminescentGrove],
                    seed: settings.seed,
                    ..Default::default()
                },
            }
        } else {
            settings.caustics = false;
            Self {
                active: Lab::Multiworld,
                other: glass_settings(settings.seed),
            }
        }
    }

    pub fn switch(&mut self, next: Lab, settings: &mut Settings) -> bool {
        if self.active == next {
            return false;
        }
        std::mem::swap(settings, &mut self.other);
        self.active = next;
        true
    }

    pub fn views(&self) -> &[View] {
        match self.active {
            Lab::Multiworld => &View::ALL,
            Lab::Glass => &[View::Radiance, View::Bounces, View::Normals, View::Samples],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_labs_preserves_independent_cameras_and_transport() {
        let mut settings = Settings {
            mode: Mode::Persistent,
            neural_cache: true,
            spatial: true,
            ..Default::default()
        };
        let multiworld = settings.clone();
        let mut labs = Labs::new(&mut settings);
        assert!(labs.switch(Lab::Glass, &mut settings));
        assert!(settings.caustics_active() && settings.denoise);
        assert!(settings.photon_guiding);
        assert!(!settings.photon_reuse && !settings.manifold_guiding);
        assert!(!settings.neural_cache && !settings.spatial);
        assert_eq!(settings.sampling, SamplingMode::Uniform);
        settings.yaw = 2.;
        settings.glass[0] = GlassPreset::Emerald;
        let glass = settings.clone();
        assert!(labs.switch(Lab::Multiworld, &mut settings));
        assert_eq!(settings, multiworld);
        assert!(labs.switch(Lab::Glass, &mut settings));
        assert_eq!(settings, glass);
        assert!(!labs.switch(Lab::Glass, &mut settings));
        assert!(!labs.views().contains(&View::Switches));
    }

    #[test]
    fn fixed_world_caustic_cli_opens_glass_without_inactive_features() {
        let mut settings = Settings {
            mode: Mode::Blend,
            caustics: true,
            neural_cache: true,
            primary_a: true,
            spatial: true,
            ..Default::default()
        };
        let labs = Labs::new(&mut settings);
        assert_eq!(labs.active, Lab::Glass);
        assert_eq!(settings.mode, Mode::Blend);
        assert!(!settings.neural_cache && !settings.primary_a && !settings.spatial);
    }
}
