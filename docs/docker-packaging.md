# Docker target

The intended product is a Docker image with baked-in weights, a browser UI and
APIs for converting and serving processed audio. This cleanup milestone does
not implement those surfaces.

When adding the image, use only the current inference package, pin and verify
the baked assets, retain model/dependency notices, and review the exact base
image and native libraries. Satisfy applicable LGPL source/replacement and NVIDIA
redistribution requirements, then validate offline inference from a fresh volume.
Source licensing is documented in [licensing.md](licensing.md).
