# data/

Generated datasets. The directory is gitignored apart from manifest
YAMLs, this README, and the `data/literature/` subtree (Phase 4 adds
the curated literature database for the verifier).

Phase 1 generates `data/synthetic_lj_v1/` on the remote and writes a
`manifest.yaml` next to the HDF5 store. The manifest is the only file
in this directory that gets committed back to the repo.

Send raw datasets between machines through the appropriate transfer
channel (rsync, scp, or shared object storage). Do not commit them.
