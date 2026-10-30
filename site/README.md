# Project website

The overview renders the root README directly. Documentation and the listening
comparison at `/examples/` share the Hangry Labs header, background, section
navigation and code-copy controls from faster-qwen3-tts-studio.

## Local preview

With Docker, Python/uv and PowerShell available:

```powershell
task pages
```

Open <http://127.0.0.1:9005/Sidon/>. `task pages-build` builds without starting a
server; `task pages-test` checks staging and routing (Node.js is needed for routing
tests). The separate pinned Ruby/Jekyll builder does not start inference, download
weights or use a GPU. Git's public-file listing includes uncommitted edits.

## GitHub Pages

The root `_config.yml` targets `https://hangry-labs.github.io/Sidon/`, using the
repository's exact case. In GitHub Settings → Pages, select deployment from the
`main` branch and `/ (root)` folder. This uses the same branch-based setup and
Pages-compatible plugins as the sibling project. Local validation does not publish
the site or change repository settings. GitHub manages its hosted build environment.

`url` is the production origin; `baseurl` is the project path. Content is English
only. The shared router supports adding actual translated READMEs later, and
hides the language selector until translations exist.

## Content and attribution

Maintain the overview in `README.md`, guides in `docs/`, and comparison in
`examples/index.md`. Markdown links resolve to rendered guides; excluded source
files open on GitHub. Use `<details markdown="1">` for collapsible Markdown.
Private/editor/runtime/build files are excluded from the staged or rendered site.

The operator-supplied before/after recordings are encoded as Opus in
`examples/audio/`, using identical libopus 96 kbps VBR / audio settings for both
sides. Its `manifest.json` records source and encoded SHA256 values, sizes,
sample rates, durations and encoder identity. Keep it aligned with deliberate
media changes. The gallery does not normalize volume, infer restoration settings
or claim a controlled perceptual benchmark.

The local reusable theme is copied from Hangry Labs faster-qwen3-tts-studio under
Apache-2.0; its full original license is in `theme/LICENSE`. Staging, preview and
build helpers retain the sibling project's MIT attribution. Theme assets are
local; no external CSS, scripts or fonts are needed.
