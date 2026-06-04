# Lifecycle

This contract describes who owns the backend, when it starts, and what closes it.

## Local VS Code CLI

- `arrayview file.npy` from a local VS Code CLI terminal should return to the prompt.
- The backend should stay up while there are open viewer tabs.
- Closing the last tab should stop the backend.
- If multiple local CLI launches happen for the same machine session, they should share one backend and separate tabs.
- Closing one tab should free only that tab's array/session, not the shared backend.

## VS Code file click and custom editor

- When opened from a file click or custom editor, prefer a direct extension-owned subprocess.
- If that path is available, do not require localhost or a shared port.
- Closing the tab should terminate the subprocess.
- This path should be transient and owned by the extension session, not by a long-lived server.

## Plain Python script

- `view(arr)` from a script should survive the script exiting.
- The backend must outlive the caller until viewer instances close.
- When the last viewer instance closes, free the arrays and shut the backend down.

## Jupyter

- Jupyter should keep the backend kernel-owned.
- An iframe disappearing should not hard-kill the backend.
- Explicit close or cleanup should free the session.
- The kernel remains the owner until the notebook session is done or cleanup runs.

## Remote and tunnel

- Remote or tunnel launches may persist when `--serve` or direct-server constraints require it.
- In these cases, persistence is acceptable as long as it matches the transport and launch mode.

## SSH

- Plain SSH should print or use a localhost port-forward.
- The user is expected to forward the port locally.
- The connection is transient: closing the viewer should end the session unless a shared server was explicitly requested.

## Shared rules

- A backend can be shared when the launch mode is local CLI or another shared-server mode.
- A backend should be transient when the owning environment is already a short-lived subprocess or extension-owned session.
- Closing the last active viewer should release arrays and any backend that is not meant to persist.
- Explicit cleanup wins over implicit disappearance.
