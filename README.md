# <img src="docs/logo.png" height="36"> arrayview

_This is my array viewer. There are many like it, but this one is mine._ 

Arrayview lets you scroll through multi-dimensional arrays.

Open it from the shell, from Python, Julia, or Matlab, or inside a Jupyter notebook. Use it locally or over SSH. 

If you work in VS Code, you can open arrays directly from the explorer; with Remote SSH, it works the same way.

It is meant to feel simple but there's more to it than meets the eye. 

Curious? Give it a try with
```bash
uvx arrayview your_array.npy
uvx arrayview path/to/dicom-series/
```

Press `v` for the three-plane ortho view. `Shift+3` replaces the current slice,
or all three ortho panes, with interactive 3D cutaway renders.


Check the [docs](https://oscarvanderheide.github.io/arrayview/) to learn more. 

**Warning**: Arrayview is still under active development. Things may break or change without warning.
