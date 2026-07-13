# Stack and Overlay Collections

Review many arrays with aligned image and mask patterns.

## Patient Stack

```bash
uvx arrayview --stack "data/*/T1w.nii.gz"
```

`*` matches one path segment. `**` matches recursively:

```bash
uvx arrayview --stack "data/**/*.nii.gz"
```

ArrayView sorts matches and adds a collection index dimension. Use `--dry-run`
to inspect pairings without opening the viewer:

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" --dry-run
```

## MR Contrasts

```bash
uvx arrayview --stack \
  "data/*/T1w.nii.gz" \
  "data/*/T2w.nii.gz" \
  "data/*/FLAIR.nii.gz"
```

Each positional pattern becomes a contrast dimension. The first `T1w`, first
`T2w`, and first `FLAIR` belong to patient 1; the second match from each pattern
belongs to patient 2.

For ambiguous layouts, define the patient key:

```bash
uvx arrayview --stack "data/**/*.nii.gz" \
  --case-regex "(?P<case>sub-[0-9]+)"
```

## Ragged Shapes

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" --stack-policy ragged
```

`--stack-policy auto` uses a dense stack when shapes match and ragged mode when
they do not. `--stack-policy dense` requires matching shapes.

## One Overlay

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" \
  --overlay "lesion=data/*/masks/lesion.nii.gz"
```

Use `name=pattern` to control the overlay label. Without `name=`, the filename
stem is used.

## Multiple Overlays

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" \
  --overlay "gt=data/*/masks/gt.nii.gz" \
  --overlay "pred=data/*/masks/pred.nii.gz"
```

Overlays are paired to cases by sorted match order unless `--case-regex` is
provided.

## Overlay Directories

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" \
  --overlay-dir "data/*/masks"
```

`--overlay-dir` creates one overlay role per mask filename inside each matched
directory. Missing masks are shown as empty overlays.

Repeat it to combine mask folders:

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" \
  --overlay-dir "data/*/manual_masks" \
  --overlay-dir "data/*/model_masks"
```

## Complete Example

```text
data/
  sub-001/
    anat/T1w.nii.gz
    anat/T2w.nii.gz
    masks/lesion.nii.gz
    masks/edema.nii.gz
  sub-002/
    anat/T1w.nii.gz
    anat/T2w.nii.gz
    masks/lesion.nii.gz
```

```bash
uvx arrayview --stack \
  "data/*/anat/T1w.nii.gz" \
  "data/*/anat/T2w.nii.gz" \
  --overlay-dir "data/*/masks" \
  --stack-policy auto \
  --dry-run
```

Remove `--dry-run` to open the viewer.

## Compressed NIfTI Performance

`.nii.gz` files are lazy through nibabel, but gzip compression limits random
slice access. Large compressed collections can feel slower when scrolling across
many patients or contrasts.

For repeated review, prefer uncompressed `.nii`, chunked `.zarr`, or a local
fast disk cache. Use `--load eager` only for small collections that fit in RAM:

```bash
uvx arrayview --stack "data/*/T1w.nii.gz" --load eager
```
