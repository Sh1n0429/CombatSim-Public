# Third-party assets and licenses

The root MIT LICENSE applies to CombatSim's own code, documentation, and original audio.
It does not replace the licenses of bundled third-party assets.

| Asset | Creator / source | License |
|---|---|---|
| F-16 mesh, `combatsim/web/arena-assets/f16.glb` | iedalton, original Sketchfab model | CC BY 4.0 |
| Three.js and official addons | Three.js authors, three@0.187.0 | MIT |
| Meshoptimizer decoder | Arseny Kapoulkine and contributors | MIT |
| `arena-score.ogg` | Original CombatSim composition, rendered with FluidSynth and GeneralUser GS | CombatSim original recording under the root MIT license; production sources described in the asset notice |
| Engine, launch, impact and other original effects | Synthesized for CombatSim | Root MIT license |

The full [asset notice](combatsim/web/arena-assets/NOTICE.txt) identifies the F-16 creator,
original URL, rendering changes, audio production sources, and vendor versions.
Retain this notice and the [CC BY 4.0 license](combatsim/web/arena-assets/CC-BY-4.0.txt)
when redistributing the aircraft asset. The [vendor MIT license](combatsim/web/arena-assets/LICENSE)
must also be retained.

Python dependencies are installed from their locked distributions rather than
vendored here; each dependency retains its own license. Source provenance is in [SOURCE.md](SOURCE.md).
