# checkpoints/

Trained model weights. The directory is gitignored apart from this
README. Checkpoints stay on the remote machine.

Each FM training run writes under `checkpoints/<fm_name>/<run_id>/`
along with a `manifest.yaml` that records the training inputs,
configuration, and software versions.
