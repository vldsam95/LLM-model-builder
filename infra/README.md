# Local educational deployment

The web application defaults to `127.0.0.1`. Run `scripts/run_web.sh` and open
http://127.0.0.1:7860. A fresh clone has no trained checkpoints; use the training
instructions in the main README or the builder's From Scratch Tiny mode.

The original host-specific Nginx, tunnel, service and runtime snapshots are not
part of this public source release. They are not needed to learn or run the project.
This repository does not provide a hardened internet-facing hosting configuration.

The educational UI supports file/URL imports and local training operations and
does not implement application authentication. Keep it local or behind an
independently configured authenticated access boundary. Run under an unprivileged
account; do not expose a privileged training service directly to the internet.
