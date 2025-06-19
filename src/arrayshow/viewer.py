import sys
from functools import partial

import numpy as np
from numpy import fft
from vispy import scene
from vispy.color import get_colormap

# Import qmricolors to register custom colormaps
try:
    import qmricolors
    # Test if the colormaps are actually available
    try:
        get_colormap('lipari')
        get_colormap('navia')
        QMRI_AVAILABLE = True
    except (KeyError, Exception):
        QMRI_AVAILABLE = False
        print("qMRI Colors: Warning - colormaps not properly registered, falling back to default colormaps")
except ImportError:
    QMRI_AVAILABLE = False

# --- Import PyQt5 directly ---
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    print(
        "Error: This application requires PyQt5. Please install it (`pip install pyqt5`)."
    )
    sys.exit(1)


class ClickableLabel(QtWidgets.QLabel):
    """A QLabel that emits a clicked signal."""

    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class NDArrayViewer(QtWidgets.QMainWindow):
    """
    A high-performance N-dimensional array viewer with fully configurable dimension roles,
    animation playback, and FFT analysis.
    """

    def __init__(self, data, title="N-D Array Viewer"):
        super().__init__()
        self.original_title = title

        # --- 1. Data and State Management ---
        # Convert non-numpy arrays to numpy (e.g. arrays from Julia, .mat files, etc)
        data = np.asarray(data)

        # Check if data is complex
        self.is_complex_data = np.iscomplexobj(data)
        self.complex_view_mode = "magnitude"  # magnitude, phase, real, imag
        
        if self.is_complex_data:
            # Store original complex data
            self.complex_data = data
            # Start with magnitude view
            data = np.abs(data).astype(np.float32)
        else:
            self.complex_data = None
            if data.dtype != np.float32:
                data = data.astype(np.float32)

        if data.ndim < 2:
            raise ValueError(
                f"Input data must have at least 2 dimensions, but got {data.ndim}."
            )

        self.data = data
        self.dims = []
        self.is_playing = False
        self.is_fft_view = False
        self.fft_data = None
        self.fft_dims = None
        self._assign_initial_roles()

        # --- 2. Create Palettes and Styles ---
        self.default_palette = QtWidgets.QApplication.instance().palette()
        self.green_palette = QtGui.QPalette(self.default_palette)
        self.green_palette.setColor(
            QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#57E079")
        )
        self.view_button_style_active = "background-color: #57E079; color: black; border: 1px solid #3A9; border-radius: 3px;"
        self.view_button_style_inactive = "background-color: #555; color: #ccc; border: 1px solid #666; border-radius: 3px;"

        # --- 3. Create VisPy Canvas ---
        self.canvas = scene.SceneCanvas(keys="interactive", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.canvas.events.key_press.connect(self._on_key_press)

        initial_slice_2d = self._get_display_image()
        self.current_colormap = "lipari" if QMRI_AVAILABLE else "grays"
        try:
            initial_cmap = get_colormap(self.current_colormap)
        except Exception:
            # Fallback to grays if initial colormap fails
            self.current_colormap = "grays"
            initial_cmap = get_colormap("grays")
        self.image = scene.visuals.Image(
            initial_slice_2d, cmap=initial_cmap, parent=self.view.scene, clim="auto"
        )
        self.image.transform = scene.transforms.STTransform()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range(x=(0, 1), y=(0, 1), margin=0)

        # --- 4. Create PyQt Widgets ---
        playback_container = QtWidgets.QWidget()
        playback_layout = QtWidgets.QHBoxLayout(playback_container)
        playback_layout.setContentsMargins(5, 5, 5, 5)
        self.play_stop_button = QtWidgets.QPushButton("Play")
        self.loop_checkbox = QtWidgets.QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)
        self.autoscale_checkbox = QtWidgets.QCheckBox("Autoscale on Scroll")
        self.fps_spinbox = QtWidgets.QDoubleSpinBox()
        self.fps_spinbox.setDecimals(1)
        self.fps_spinbox.setRange(0.1, 100.0)
        self.fps_spinbox.setValue(10.0)
        self.fps_spinbox.setSuffix(" FPS")
        
        # Complex array controls
        self.complex_view_combo = QtWidgets.QComboBox()
        self.complex_view_combo.addItems(["Magnitude", "Phase", "Real", "Imaginary"])
        self.complex_view_combo.setCurrentText("Magnitude")
        self.complex_view_combo.currentTextChanged.connect(self._on_complex_view_changed)
        
        # Colormap controls
        self.colormap_combo = QtWidgets.QComboBox()
        colormap_options = ["grays", "hot", "viridis", "coolwarm"]
        if QMRI_AVAILABLE:
            colormap_options = ["lipari", "navia"] + colormap_options
        self.colormap_combo.addItems(colormap_options)
        # Set dropdown to match the actual colormap being used
        self.colormap_combo.setCurrentText(self.current_colormap)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        
        playback_layout.addWidget(self.play_stop_button)
        playback_layout.addWidget(self.loop_checkbox)
        playback_layout.addWidget(self.autoscale_checkbox)
        
        # Add complex controls if data is complex
        if self.is_complex_data:
            playback_layout.addWidget(QtWidgets.QLabel("Complex View:"))
            playback_layout.addWidget(self.complex_view_combo)
        
        # Add colormap controls
        playback_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        playback_layout.addWidget(self.colormap_combo)
        
        playback_layout.addStretch()
        playback_layout.addWidget(QtWidgets.QLabel("Speed:"))
        playback_layout.addWidget(self.fps_spinbox)

        controls_container = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QGridLayout(controls_container)
        self.sliders = []
        self.view_buttons = []
        self.info_labels = []
        for i, dim in enumerate(self.dims):
            label_widget = QtWidgets.QWidget()
            label_layout = QtWidgets.QHBoxLayout(label_widget)
            label_layout.setContentsMargins(0, 0, 5, 0)
            label_layout.setSpacing(5)
            label_layout.addWidget(QtWidgets.QLabel(f"Dim {i}:"))
            x_button = ClickableLabel("[X]")
            x_button.setFixedSize(20, 20)
            x_button.setAlignment(QtCore.Qt.AlignCenter)
            x_button.clicked.connect(
                partial(self._set_view_dimension_from_button, i, 0)
            )
            y_button = ClickableLabel("[Y]")
            y_button.setFixedSize(20, 20)
            y_button.setAlignment(QtCore.Qt.AlignCenter)
            y_button.clicked.connect(
                partial(self._set_view_dimension_from_button, i, 1)
            )
            g_button = ClickableLabel("[G]")
            g_button.setFixedSize(20, 20)
            g_button.setAlignment(QtCore.Qt.AlignCenter)
            g_button.clicked.connect(partial(self._toggle_grid_dimension, i))
            label_layout.addWidget(x_button)
            label_layout.addWidget(y_button)
            label_layout.addWidget(g_button)
            self.view_buttons.append((x_button, y_button, g_button))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(dim["size"] - 1)
            slider.sliderPressed.connect(partial(self._on_slider_pressed, i))
            slider.valueChanged.connect(partial(self._on_slider_moved, i))
            self.sliders.append(slider)
            info_label = QtWidgets.QLabel()
            info_label.setMinimumWidth(80)
            self.info_labels.append(info_label)
            self.controls_layout.addWidget(label_widget, i, 0)
            self.controls_layout.addWidget(slider, i, 1)
            self.controls_layout.addWidget(info_label, i, 2)
        self.controls_layout.setColumnStretch(1, 1)

        # --- 5. Arrange Main Layout ---
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.addWidget(self.canvas.native)
        main_layout.addWidget(playback_container)
        main_layout.addWidget(controls_container)
        self.setCentralWidget(central_widget)
        self.setWindowTitle(self.original_title)
        self.resize(700, 750)

        # --- 6. Setup Timer and Finalize ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._advance_slice)
        self.play_stop_button.clicked.connect(self._toggle_playback)
        self.fps_spinbox.valueChanged.connect(self._update_timer_interval)
        self._update_view()

    @property
    def active_data(self):
        if self.is_fft_view:
            return self.fft_data
        elif self.is_complex_data:
            return self._get_complex_view_data()
        else:
            return self.data
    
    def _get_complex_view_data(self):
        """Get the appropriate view of complex data based on current mode."""
        if not self.is_complex_data or self.complex_data is None:
            return self.data
            
        if self.complex_view_mode == "magnitude":
            return np.abs(self.complex_data).astype(np.float32)
        elif self.complex_view_mode == "phase":
            return np.angle(self.complex_data).astype(np.float32)
        elif self.complex_view_mode == "real":
            return np.real(self.complex_data).astype(np.float32)
        elif self.complex_view_mode == "imag":
            return np.imag(self.complex_data).astype(np.float32)
        else:
            return np.abs(self.complex_data).astype(np.float32)
    
    def _on_complex_view_changed(self, text):
        """Handle complex view mode changes from dropdown."""
        mode_map = {
            "Magnitude": "magnitude",
            "Phase": "phase",
            "Real": "real",
            "Imaginary": "imag"
        }
        self.complex_view_mode = mode_map.get(text, "magnitude")
        self.data = self._get_complex_view_data()
        self.image.clim = "auto"
        self._update_view()
    
    def _cycle_complex_view(self):
        """Cycle through complex view modes with 'c' key."""
        if not self.is_complex_data:
            return
            
        modes = ["magnitude", "phase", "real", "imag"]
        current_idx = modes.index(self.complex_view_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.complex_view_mode = modes[next_idx]
        
        # Update dropdown to match
        mode_names = ["Magnitude", "Phase", "Real", "Imaginary"]
        self.complex_view_combo.setCurrentText(mode_names[next_idx])
        
        self.data = self._get_complex_view_data()
        self.image.clim = "auto"
        self._update_view()
    
    def _on_colormap_changed(self, text):
        """Handle colormap changes from dropdown."""
        try:
            self.current_colormap = text.lower()
            new_cmap = get_colormap(self.current_colormap)
            self.image.cmap = new_cmap
            self.image.update()
        except Exception as e:
            print(f"Warning: Could not set colormap '{text}': {e}")
            # Fallback to grays if colormap fails
            try:
                self.current_colormap = "grays"
                fallback_cmap = get_colormap("grays")
                self.image.cmap = fallback_cmap
                self.image.update()
            except Exception:
                pass

    def _assign_initial_roles(self):
        self.dims.clear()
        for i, size in enumerate(self.data.shape):
            role = "fixed"
            if i == 0:
                role = "view_x"
            elif i == 1:
                role = "view_y"
            elif i == 2 and self.data.ndim > 2:
                role = "scroll"
            self.dims.append({"role": role, "index": size // 2, "size": size})
        self._update_scroll_dim_index()

    def _get_display_image(self):
        grid_dim_idx = next(
            (i for i, d in enumerate(self.dims) if d["role"] == "view_grid"), None
        )
        if grid_dim_idx is None:
            return self._get_single_slice()
        else:
            return self._build_grid_image(grid_dim_idx)

    def _get_single_slice(self, slicer_override=None):
        slicer = slicer_override or [
            slice(None) if "view" in d["role"] else d["index"] for d in self.dims
        ]
        x_dim_idx = next(i for i, d in enumerate(self.dims) if d["role"] == "view_x")
        y_dim_idx = next(i for i, d in enumerate(self.dims) if d["role"] == "view_y")
        slice_data = self.active_data[tuple(slicer)]
        if y_dim_idx < x_dim_idx:
            slice_data = slice_data.T
        return slice_data

    def _calculate_grid_layout(self, num_images):
        if num_images == 0:
            return 0, 0
        best_rows = 1
        for r in range(int(np.sqrt(num_images)), 0, -1):
            if num_images % r == 0:
                best_rows = r
                break
        return best_rows, num_images // best_rows

    def _build_grid_image(self, grid_dim_idx):
        grid_dim = self.dims[grid_dim_idx]
        num_images = grid_dim["size"]
        slicer = [slice(None) if "view" in d["role"] else d["index"] for d in self.dims]
        slicer[grid_dim_idx] = 0
        template_slice = self._get_single_slice(slicer)
        slice_h, slice_w = template_slice.shape
        rows, cols = self._calculate_grid_layout(num_images)
        grid_image = np.zeros(
            (rows * slice_h, cols * slice_w), dtype=self.active_data.dtype
        )
        for i in range(num_images):
            slicer[grid_dim_idx] = i
            current_slice = self._get_single_slice(slicer)
            r, c = i // cols, i % cols
            grid_image[
                r * slice_h : (r + 1) * slice_h, c * slice_w : (c + 1) * slice_w
            ] = current_slice
        return grid_image

    def _reassign_scroll_and_fixed(self):
        found_scroll = False
        for dim in self.dims:
            if "view" not in dim["role"]:
                if not found_scroll:
                    dim["role"] = "scroll"
                    found_scroll = True
                else:
                    dim["role"] = "fixed"
        self._update_scroll_dim_index()

    def _reassign_all_roles(self, x_idx, y_idx):
        if self.is_playing:
            self._toggle_playback()
        for i, dim in enumerate(self.dims):
            if i == x_idx:
                dim["role"] = "view_x"
            elif i == y_idx:
                dim["role"] = "view_y"
            else:
                dim["role"] = "fixed"
        self._reassign_scroll_and_fixed()
        self.image.clim = "auto"
        self._update_view()

    def _set_view_dimension_from_button(self, new_dim_idx, axis):
        current_x_idx = next(
            i for i, d in enumerate(self.dims) if d["role"] == "view_x"
        )
        current_y_idx = next(
            i for i, d in enumerate(self.dims) if d["role"] == "view_y"
        )
        if (axis == 0 and new_dim_idx == current_y_idx) or (
            axis == 1 and new_dim_idx == current_x_idx
        ):
            self._reassign_all_roles(current_y_idx, current_x_idx)
        else:
            if axis == 0:
                self._reassign_all_roles(new_dim_idx, current_y_idx)
            else:
                self._reassign_all_roles(current_x_idx, new_dim_idx)

    def _toggle_grid_dimension(self, dim_idx):
        if self.is_playing:
            self._toggle_playback()
        current_role = self.dims[dim_idx]["role"]
        if current_role == "view_grid":
            self.dims[dim_idx]["role"] = "scroll"
            self._reassign_scroll_and_fixed()
        elif "view" not in current_role:
            for d in self.dims:
                if d["role"] == "view_grid":
                    d["role"] = "fixed"
            self.dims[dim_idx]["role"] = "view_grid"
            self._reassign_scroll_and_fixed()
        self.image.clim = "auto"
        self._update_view()

    def _prompt_for_view_dims(self):
        if self.is_playing:
            self._toggle_playback()
        current_x = next(i for i, d in enumerate(self.dims) if d["role"] == "view_x")
        current_y = next(i for i, d in enumerate(self.dims) if d["role"] == "view_y")
        prompt_text = f"Enter two different dimension indices (0-{self.data.ndim - 1}) separated by a space.\nCurrent is X={current_x}, Y={current_y}."
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Set View Dimensions", prompt_text
        )
        if not (ok and text):
            return
        try:
            parts = text.split()
            if len(parts) != 2:
                raise ValueError("Please enter exactly two numbers.")
            x_dim, y_dim = int(parts[0]), int(parts[1])
            if not (0 <= x_dim < self.data.ndim and 0 <= y_dim < self.data.ndim):
                raise ValueError(
                    f"Dimensions must be between 0 and {self.data.ndim - 1}."
                )
            if x_dim == y_dim:
                raise ValueError("The two dimensions must be different.")
            self._reassign_all_roles(x_dim, y_dim)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(e))

    def _update_scroll_dim_index(self):
        self.scroll_dim_idx = next(
            (i for i, d in enumerate(self.dims) if d["role"] == "scroll"), -1
        )

    def _set_scroll_dimension(self, new_scroll_idx):
        if self.is_playing:
            self._toggle_playback()
        if "view" in self.dims[new_scroll_idx]["role"]:
            return
        old_scroll_idx = self.scroll_dim_idx
        if old_scroll_idx == new_scroll_idx:
            return
        if old_scroll_idx != -1:
            self.dims[old_scroll_idx]["role"] = "fixed"
        self.dims[new_scroll_idx]["role"] = "scroll"
        self._update_scroll_dim_index()
        self._update_view()

    def _change_scroll_dimension_by_cycle(self, direction):
        if self.is_playing:
            self._toggle_playback()
        cyclable_dims = [
            (i, d) for i, d in enumerate(self.dims) if "view" not in d["role"]
        ]
        if len(cyclable_dims) < 2:
            return
        cyclable_indices = [i for i, d in cyclable_dims]
        try:
            current_pos = cyclable_indices.index(self.scroll_dim_idx)
            new_pos = (current_pos + (1 if direction == "next" else -1)) % len(
                cyclable_indices
            )
            self._set_scroll_dimension(cyclable_indices[new_pos])
        except ValueError:
            self._set_scroll_dimension(cyclable_indices[0])

    def _on_key_press(self, event):
        if event.key in ("j", "k", "l", "h", "v", "g", "f", "c"):
            if self.is_playing:
                self._toggle_playback()

        if self.scroll_dim_idx != -1 and event.key in ("j", "k"):
            current_val = self.dims[self.scroll_dim_idx]["index"]
            if event.key == "j":
                max_val = self.dims[self.scroll_dim_idx]["size"] - 1
                self.dims[self.scroll_dim_idx]["index"] = min(max_val, current_val + 1)
            else:
                self.dims[self.scroll_dim_idx]["index"] = max(0, current_val - 1)
            self._update_view()
        elif event.key == "l":
            self._change_scroll_dimension_by_cycle("next")
        elif event.key == "h":
            self._change_scroll_dimension_by_cycle("prev")
        elif event.key == "v":
            self._prompt_for_view_dims()
        elif event.key == "g":
            grid_dim_idx = next(
                (i for i, d in enumerate(self.dims) if d["role"] == "view_grid"), None
            )
            if grid_dim_idx is not None:
                self._toggle_grid_dimension(grid_dim_idx)
            elif self.scroll_dim_idx != -1:
                self._toggle_grid_dimension(self.scroll_dim_idx)
        elif event.key == "f":
            self._toggle_fft_view()
        elif event.key == "c":
            self._cycle_complex_view()

    def _on_slider_pressed(self, dim_idx):
        if self.is_playing:
            self._toggle_playback()
        if self.dims[dim_idx]["role"] == "fixed":
            self._set_scroll_dimension(dim_idx)

    def _on_slider_moved(self, dim_idx, value):
        if self.is_playing:
            self._toggle_playback()
        if "view" not in self.dims[dim_idx]["role"]:
            self.dims[dim_idx]["index"] = value
            self._update_view()

    def _update_ui_state(self):
        if self.is_fft_view:
            self.setWindowTitle(
                f"{self.original_title} - FFT View (dims: {self.fft_dims})"
            )
        elif self.is_complex_data:
            self.setWindowTitle(
                f"{self.original_title} - Complex View ({self.complex_view_mode.capitalize()})"
            )
        else:
            self.setWindowTitle(self.original_title)

        for i, dim in enumerate(self.dims):
            slider = self.sliders[i]
            x_button, y_button, g_button = self.view_buttons[i]
            info_label = self.info_labels[i]
            slider.blockSignals(True)
            slider.setValue(dim["index"])
            slider.blockSignals(False)
            is_view = "view" in dim["role"]
            slider.setEnabled(not is_view)
            x_button.setStyleSheet(
                self.view_button_style_active
                if dim["role"] == "view_x"
                else self.view_button_style_inactive
            )
            y_button.setStyleSheet(
                self.view_button_style_active
                if dim["role"] == "view_y"
                else self.view_button_style_inactive
            )
            g_button.setStyleSheet(
                self.view_button_style_active
                if dim["role"] == "view_grid"
                else self.view_button_style_inactive
            )
            if dim["role"] == "scroll":
                slider.setPalette(self.green_palette)
            else:
                slider.setPalette(self.default_palette)
            if is_view:
                info_label.setText(f": / {dim['size']}")
            else:
                info_label.setText(f"{dim['index'] + 1} / {dim['size']}")

    def _update_view(self):
        """Central function to update the rendered image and all UI components."""
        display_image = self._get_display_image()
        h, w = display_image.shape
        if h > 0 and w > 0:
            self.image.transform.scale = (1 / w, 1 / h, 1)

        self.image.set_data(display_image)

        if self.autoscale_checkbox.isChecked():
            self.image.clim = "auto"

        # **FIX:** Explicitly tell VisPy to redraw the canvas.
        # This was previously only happening implicitly when clim was set.
        self.image.update()

        self._update_ui_state()

    # --- Playback Methods ---
    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            if self.scroll_dim_idx == -1:
                self.is_playing = False
                return
            self._update_timer_interval()
            self.timer.start()
            self.play_stop_button.setText("Stop")
        else:
            self.timer.stop()
            self.play_stop_button.setText("Play")

    def _update_timer_interval(self):
        fps = self.fps_spinbox.value()
        if fps > 0:
            self.timer.setInterval(int(1000 / fps))

    def _advance_slice(self):
        if self.scroll_dim_idx == -1:
            self._toggle_playback()
            return

        scroll_dim = self.dims[self.scroll_dim_idx]
        current_index = scroll_dim["index"]
        max_index = scroll_dim["size"] - 1
        new_index = current_index + 1

        if new_index > max_index:
            if self.loop_checkbox.isChecked():
                new_index = 0
            else:
                self._toggle_playback()
                return

        scroll_dim["index"] = new_index
        self._update_view()

    # --- FFT Methods ---
    def _toggle_fft_view(self):
        if self.is_fft_view:
            self.is_fft_view = False
            self.fft_data = None
            self.fft_dims = None
        else:
            self._prompt_for_fft_dims()
            if not self.is_fft_view:
                return

        self.image.clim = "auto"
        self._update_view()

    def _prompt_for_fft_dims(self):
        prompt_text = f"Enter dimension indices (0-{self.data.ndim - 1}) for FFT, separated by spaces."
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Set FFT Dimensions", prompt_text
        )
        if not (ok and text):
            return

        try:
            parts = text.split()
            if not parts:
                raise ValueError("Please enter at least one dimension.")
            dims_to_fft = sorted(list(set([int(p) for p in parts])))
            for d in dims_to_fft:
                if not (0 <= d < self.data.ndim):
                    raise ValueError(
                        f"Dimension {d} is out of range (0-{self.data.ndim - 1})."
                    )
            self._compute_fft(dims_to_fft)
            self.is_fft_view = True
            self.fft_dims = dims_to_fft
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(e))
            self.is_fft_view = False

    def _compute_fft(self, dims_to_fft):
        print(f"Computing FFT on dimensions: {dims_to_fft}...")
        fft_result = fft.fftn(self.data, axes=dims_to_fft)
        shifted_fft = fft.fftshift(fft_result, axes=dims_to_fft)
        magnitude = np.abs(shifted_fft)
        log_magnitude = np.log1p(magnitude)
        self.fft_data = log_magnitude.astype(np.float32)
        print("FFT computation complete.")
