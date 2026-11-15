# Mesh sources and attribution

These model-data terms are separate from the renderer's MIT source license. Both meshes are embedded in the executable; the app does not fetch assets at runtime.

## Modified Utah Teapot (2026)

- Origin: the historic **Utah Teapot**, Martin Newell, University of Utah; 2026 interior, trimming and continuity work by Cem Yuksel.
- Source and model-use terms: https://graphics.cs.utah.edu/teapot/
- Exported on 2026-11-14 from the official site's triangle-mesh generator, resolution 6, with the 2026 interior, round bottom, circular profile, trimmed/chamfered joins, curvature continuity, symmetric triangulation, both sides, Blinn scaling, and Y-up orientation.
- Generator options bitmask: `227581951`. The generator and WebAssembly URLs are `https://graphics.cs.utah.edu/teapot/teapot_generator.js` and `https://graphics.cs.utah.edu/teapot/teapot_generator.wasm`.
- Adaptation: retained positions and triangle winding; welded duplicate positions within 1e-6 world units; discarded unused normal/UV streams. No boundary caps were needed. Stored as `meshes/utah-teapot-2026.obj` (5,024 triangles).
- The University of Utah permits use of its model files, including commercial use, and requests preservation of the Utah Teapot identity and source attribution. See the source page for the complete terms and warranty disclaimer.

## Modified Suzanne

- Source: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Suzanne
- Copyright 2017 UX3D; Norbert Nopper. Creative Commons Zero v1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/
- Source files: `Models/Suzanne/glTF/Suzanne.gltf` and `Suzanne.bin` from that repository, retrieved 2026-11-14.
- Adaptation: extracted triangle positions/indices from glTF, welded duplicate positions within 1e-6, and closed four boundary loops with consistently wound triangle fans to support volume entry/exit. Stored as `meshes/suzanne.obj` (4,020 triangles after closure).

Both bundled meshes are tested for non-degenerate triangles and two faces per undirected edge. The renderer deliberately uses geometric face normals for faceted glass. No upstream generator code or rendering-library binaries are bundled.
