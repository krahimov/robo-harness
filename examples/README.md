# Reproducible examples

Start the app using `python3 scripts/dev.py`.

Run `python3 scripts/smoke.py` in another terminal. It records two scripted demonstrations at different starting positions, compares one versus two demonstrations over matched evaluation seeds, and writes the complete comparison bundle under `.data/smoke/`. The model is trajectory replay. This checks the experiment workflow without claiming learned behavior or making a provider call.

For photo-guided authoring, upload a factory photo in Scene builder and use `factory-brief.txt` as a starting brief. Alternatively, run `python3 scripts/smoke.py --author-photo /absolute/path/photo.jpg`. This explicitly calls the selected author connection and consumes its usage. The script defaults to the Codex connection; use `--provider openai` for an API key configured on the backend.

The public reference used in development is [Conveyor system in a factory](https://commons.wikimedia.org/wiki/File:Conveyor_system_in_a_factory.jpg), photographed by falco, published on Pixabay in 2016, and listed on Commons under CC0 1.0. The original is not bundled. Use your own photo or obtain it from the source page. Photos alone do not establish dimensions, mass, friction, or safe robot operating conditions. The supplied brief records its tabletop scale assumptions explicitly.
