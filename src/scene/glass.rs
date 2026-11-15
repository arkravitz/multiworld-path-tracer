use clap::ValueEnum;
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum GlassPreset {
    Assorted,
    Clear,
    Borosilicate,
    Flint,
    Diamond,
    Amber,
    Emerald,
    Sapphire,
    Water,
}
impl GlassPreset {
    pub const ALL: [Self; 9] = [
        Self::Assorted,
        Self::Clear,
        Self::Borosilicate,
        Self::Flint,
        Self::Diamond,
        Self::Amber,
        Self::Emerald,
        Self::Sapphire,
        Self::Water,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Assorted => "Curated glass palette",
            Self::Clear => "Clear glass",
            Self::Borosilicate => "Borosilicate",
            Self::Flint => "Flint crystal",
            Self::Diamond => "Diamond",
            Self::Amber => "Amber glass",
            Self::Emerald => "Emerald glass",
            Self::Sapphire => "Sapphire glass",
            Self::Water => "Water / low index",
        }
    }
    /// RGB transmittance through one world unit, plus 2+IOR. Assorted is a scene
    /// selection; when used directly as a material it resolves to clear glass.
    pub fn material(self) -> [f32; 4] {
        match self {
            Self::Assorted | Self::Clear => [0.995, 0.998, 0.999, 3.52],
            Self::Borosilicate => [0.998, 0.999, 0.997, 3.47],
            Self::Flint => [0.992, 0.996, 0.999, 3.62],
            Self::Diamond => [0.999, 0.999, 0.999, 4.417],
            Self::Amber => [0.99, 0.78, 0.43, 3.52],
            Self::Emerald => [0.40, 0.97, 0.64, 3.52],
            Self::Sapphire => [0.45, 0.70, 0.99, 3.52],
            Self::Water => [0.97, 0.993, 0.998, 3.333],
        }
    }
}
