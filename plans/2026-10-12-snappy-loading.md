# Goal: ArrayView opens snappily, especially from slow network drives

Owner: Oscar. Started 2026-10-12.

## What "done" means

ArrayView feels snappy in every common way of using it, and nothing that works
today breaks. The main pain is opening files from the slow SMB drive; opening
from local disk is "somewhat snappy" and should get snappier too.

## How it is measured

A stopwatch times from pressing enter until the requested array is drawn (the
viewer's first rendered frame, plus the fade-in the user actually waits for).
It launches the real CLI detached from VS Code and loads the page in a headless
browser, so it never opens anything in the user's window.

Test data — the same five files in both places:

- local: `/localscratch/oheide/tmp/arrayview_speedtest/`
- SMB: `/smb/user/oheide/DS-Data/Radiotherapie/Research/USER/oheide/arrayview_speedtest/`

| file | what it is |
|------|-----------|
| small.npy | 44 MB, (224, 240, 204) float32 |
| medium.npy | 241 MB, (224, 224, 240, 5) float32 |
| maps.nii | 289 MB, (224, 224, 240, 6) float32 |
| maps.nii.gz | 110 MB compressed version of maps.nii |
| large_complex.npy | 1.2 GB, (224, 224, 240, 12) complex64 |

Each file is timed cold (evicted from the OS cache, as on first open of the
day) and warm (just opened before).

The SMB drive reads at ~40 MB/s with one reader and ~75–90 MB/s with several
in parallel; a 4 KB random read costs ~40 ms.

## Targets

- Opening again on local disk: under 0.5 s.
- First open on local disk: under 1.5 s.
- SMB: the first picture appears long before the whole file has been read,
  wherever the file layout allows it; where it does not, the read runs at
  close to the drive's real speed and the wait is visibly explained.
- VS Code tunnel: no slower than local plus the network's own round trips.
- Scrolling slices never lags behind the hand.

## Rules

- Biggest wait first; keep a change only if the stopwatch shows it helps.
- Any change that touches opening/display is checked in the real VS Code tunnel
  window before it is kept (the owner's daily path). Ask before opening tabs in
  the user's VS Code.
- A speed-up that breaks any working setup is reverted.
- One change per commit.

## Log

- 2026-10-12: baseline recorded (see below).
