"""
Lightweight GUI for loading equilibria, plotting profiles, and visualizing
magnetic-coordinate overlays.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..coordinates.registry import list_coordinate_systems
from ..io.eqdsk import eqdsk as EQDSK


def list_plot_variables(eq) -> Dict[str, List[str]]:
    """
    Return available 1D and 2D plottable variable names for an equilibrium.

    Keys:
      - ``"1d"``
      - ``"2d"``
    """
    one_d: List[str] = []
    two_d: List[str] = []

    # Structured names
    for name in eq.field.data_vars:
        if eq.field[name].ndim == 2:
            two_d.append(f"field.{name}")
    for name in eq.flux.data_vars:
        if eq.flux[name].ndim == 2:
            two_d.append(f"flux.{name}")
    for name in eq.profiles.data_vars:
        if eq.profiles[name].ndim == 1:
            one_d.append(f"profiles.{name}")

    # Backward-compatible plotting registry names
    one_d.extend(eq.plot_1d_names.keys())
    two_d.extend(eq.plot_2d_names.keys())

    return {
        "1d": sorted(set(one_d)),
        "2d": sorted(set(two_d)),
    }


def resolve_plot_variable(eq, name: str) -> Tuple[xr.DataArray, bool]:
    """
    Resolve a plot variable name to DataArray and dimensionality flag.
    """
    resolved = eq._resolve_plot_variable(name)  # noqa: SLF001 - reusing internal resolver
    if resolved is not None:
        var, is_2d = resolved
        return var, bool(is_2d)

    if name in eq.plot_2d_names:
        return eq.plot_2d_names[name], True
    if name in eq.plot_1d_names:
        return eq.plot_1d_names[name], False

    raise ValueError(f"Unknown plot variable '{name}'.")


def sample_indices(size: int, requested: int) -> np.ndarray:
    """
    Evenly sample indices across ``[0, size-1]`` with at least two points
    when possible.
    """
    if size <= 0:
        return np.array([], dtype=int)
    if size == 1:
        return np.array([0], dtype=int)
    requested = max(2, min(size, int(requested)))
    return np.linspace(0, size - 1, requested, dtype=int)


class EquilibriumGuiApp:
    """
    Tk application controller.
    """

    def __init__(
        self,
        root,
        tk_mod,
        ttk_mod,
        filedialog_mod,
        messagebox_mod,
        FigureCls,
        FigureCanvasTkAggCls,
        NavigationToolbar2TkCls,
    ) -> None:
        self.root = root
        self.tk = tk_mod
        self.ttk = ttk_mod
        self.filedialog = filedialog_mod
        self.messagebox = messagebox_mod
        self.Figure = FigureCls
        self.FigureCanvasTkAgg = FigureCanvasTkAggCls
        self.NavigationToolbar2Tk = NavigationToolbar2TkCls

        self.eq = None
        self._computed_coords = {}
        self._active_colorbar = None
        self._current_plot_is_2d = False

        self.path_var = self.tk.StringVar()
        self.plot_mode_var = self.tk.StringVar(value="2D")
        self.plot_var_var = self.tk.StringVar()
        self.lpsi_var = self.tk.StringVar(value="201")
        self.ltheta_var = self.tk.StringVar(value="256")
        self.nsurf_var = self.tk.StringVar(value="10")
        self.status_var = self.tk.StringVar(value="Load an EQDSK file to begin.")

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.title("pyCOCOS Equilibrium Viewer")
        self.root.geometry("1300x820")

        container = self.ttk.Frame(self.root, padding=8)
        container.pack(fill=self.tk.BOTH, expand=True)

        left = self.ttk.Frame(container, width=360)
        left.pack(side=self.tk.LEFT, fill=self.tk.Y, padx=(0, 8))
        right = self.ttk.Frame(container)
        right.pack(side=self.tk.RIGHT, fill=self.tk.BOTH, expand=True)

        self._build_file_section(left)
        self._build_plot_section(left)
        self._build_coordinates_section(left)
        self._build_status_section(left)

        self.figure = self.Figure(figsize=(8.5, 6.0), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("No equilibrium loaded")
        self.ax.set_xlabel("R [m]")
        self.ax.set_ylabel("z [m]")
        self.ax.grid(True, alpha=0.25)

        self.canvas = self.FigureCanvasTkAgg(self.figure, master=right)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=self.tk.BOTH, expand=True)

        toolbar = self.NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=self.tk.BOTTOM, fill=self.tk.X)

        self.canvas.draw_idle()

    def _build_file_section(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Equilibrium", padding=8)
        frame.pack(fill=self.tk.X, pady=(0, 8))

        self.ttk.Label(frame, text="EQDSK path:").pack(anchor=self.tk.W)
        self.ttk.Entry(frame, textvariable=self.path_var).pack(fill=self.tk.X, pady=(2, 6))

        btn_row = self.ttk.Frame(frame)
        btn_row.pack(fill=self.tk.X)

        self.ttk.Button(btn_row, text="Browse", command=self.on_browse).pack(
            side=self.tk.LEFT, padx=(0, 6)
        )
        self.ttk.Button(btn_row, text="Load", command=self.on_load).pack(side=self.tk.LEFT)

    def _build_plot_section(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Variable Plotting", padding=8)
        frame.pack(fill=self.tk.X, pady=(0, 8))

        self.ttk.Label(frame, text="Plot type:").pack(anchor=self.tk.W)
        mode_combo = self.ttk.Combobox(
            frame,
            textvariable=self.plot_mode_var,
            state="readonly",
            values=["2D", "1D"],
        )
        mode_combo.pack(fill=self.tk.X, pady=(2, 6))
        mode_combo.bind("<<ComboboxSelected>>", self.on_plot_mode_changed)

        self.ttk.Label(frame, text="Variable:").pack(anchor=self.tk.W)
        self.plot_var_combo = self.ttk.Combobox(
            frame,
            textvariable=self.plot_var_var,
            state="readonly",
            values=[],
        )
        self.plot_var_combo.pack(fill=self.tk.X, pady=(2, 8))

        btn_row = self.ttk.Frame(frame)
        btn_row.pack(fill=self.tk.X)
        self.ttk.Button(btn_row, text="Plot Selected", command=self.on_plot_variable).pack(
            side=self.tk.LEFT, padx=(0, 6)
        )
        self.ttk.Button(btn_row, text="Clear Axis", command=self.on_clear_axis).pack(side=self.tk.LEFT)

    def _build_coordinates_section(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Magnetic Coordinates", padding=8)
        frame.pack(fill=self.tk.BOTH, expand=True, pady=(0, 8))

        self.ttk.Label(frame, text="Coordinate systems (multi-select):").pack(anchor=self.tk.W)

        list_frame = self.ttk.Frame(frame)
        list_frame.pack(fill=self.tk.X, pady=(2, 8))

        self.coord_listbox = self.tk.Listbox(
            list_frame,
            selectmode=self.tk.EXTENDED,
            height=6,
            exportselection=False,
        )
        self.coord_listbox.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        scroll = self.ttk.Scrollbar(list_frame, orient=self.tk.VERTICAL, command=self.coord_listbox.yview)
        scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        self.coord_listbox.configure(yscrollcommand=scroll.set)

        for idx, name in enumerate(sorted(list_coordinate_systems())):
            self.coord_listbox.insert(self.tk.END, name)
            if name == "boozer":
                self.coord_listbox.selection_set(idx)

        form = self.ttk.Frame(frame)
        form.pack(fill=self.tk.X, pady=(0, 8))

        self.ttk.Label(form, text="lpsi").grid(row=0, column=0, sticky=self.tk.W)
        self.ttk.Entry(form, textvariable=self.lpsi_var, width=8).grid(row=0, column=1, padx=(6, 12))
        self.ttk.Label(form, text="ltheta").grid(row=0, column=2, sticky=self.tk.W)
        self.ttk.Entry(form, textvariable=self.ltheta_var, width=8).grid(row=0, column=3, padx=(6, 0))

        self.ttk.Label(form, text="#surfaces").grid(row=1, column=0, sticky=self.tk.W, pady=(6, 0))
        self.ttk.Entry(form, textvariable=self.nsurf_var, width=8).grid(
            row=1, column=1, padx=(6, 12), pady=(6, 0)
        )

        btn_row = self.ttk.Frame(frame)
        btn_row.pack(fill=self.tk.X)
        self.ttk.Button(btn_row, text="Compute Selected", command=self.on_compute_coordinates).pack(
            side=self.tk.LEFT, padx=(0, 6)
        )
        self.ttk.Button(btn_row, text="Overlay on 2D", command=self.on_overlay_coordinates).pack(
            side=self.tk.LEFT
        )

    def _build_status_section(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Status", padding=8)
        frame.pack(fill=self.tk.X)
        self.ttk.Label(
            frame,
            textvariable=self.status_var,
            wraplength=320,
            justify=self.tk.LEFT,
        ).pack(anchor=self.tk.W)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _get_selected_coordinate_systems(self) -> List[str]:
        return [self.coord_listbox.get(i) for i in self.coord_listbox.curselection()]

    def _get_int_value(self, var, field_name: str) -> int:
        try:
            value = int(var.get())
        except ValueError as exc:
            raise ValueError(f"'{field_name}' must be an integer.") from exc
        if value <= 0:
            raise ValueError(f"'{field_name}' must be positive.")
        return value

    def _remove_colorbar(self) -> None:
        if self._active_colorbar is not None:
            try:
                self._active_colorbar.remove()
            except Exception:
                pass
            self._active_colorbar = None

    def _refresh_variable_list(self) -> None:
        if self.eq is None:
            self.plot_var_combo.configure(values=[])
            self.plot_var_var.set("")
            return

        vars_map = list_plot_variables(self.eq)
        key = "2d" if self.plot_mode_var.get() == "2D" else "1d"
        values = vars_map[key]
        self.plot_var_combo.configure(values=values)
        if values:
            self.plot_var_var.set(values[0])
        else:
            self.plot_var_var.set("")

    def on_plot_mode_changed(self, _event=None) -> None:
        self._refresh_variable_list()

    def on_browse(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Select EQDSK file",
            filetypes=[
                ("EQDSK files", "*.geqdsk *.eqdsk *.gfile *.g"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.path_var.set(path)

    def on_load(self) -> None:
        raw_path = self.path_var.get().strip()
        if not raw_path:
            self.messagebox.showwarning("Missing input", "Please select an EQDSK file.")
            return

        path = Path(raw_path).expanduser()
        if not path.exists():
            self.messagebox.showerror("File not found", f"Cannot locate file:\n{path}")
            return

        try:
            t0 = time.perf_counter()
            self.eq = EQDSK(str(path))
            dt = time.perf_counter() - t0
        except Exception as exc:
            self.messagebox.showerror("Load failed", f"Could not load equilibrium:\n{exc}")
            return

        self._computed_coords = {}
        self._refresh_variable_list()
        self.ax.clear()
        self._remove_colorbar()
        self.ax.set_title(f"Loaded: {path.name}")
        self.ax.set_xlabel("R [m]")
        self.ax.set_ylabel("z [m]")
        self.ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()
        self._set_status(f"Loaded {path.name} in {dt:.2f}s.")

    def on_plot_variable(self) -> None:
        if self.eq is None:
            self.messagebox.showwarning("No equilibrium", "Load an equilibrium first.")
            return

        name = self.plot_var_var.get().strip()
        if not name:
            self.messagebox.showwarning("No variable", "Select a variable to plot.")
            return

        try:
            var, is_2d = resolve_plot_variable(self.eq, name)
        except Exception as exc:
            self.messagebox.showerror("Variable error", str(exc))
            return

        self.ax.clear()
        self._remove_colorbar()

        try:
            _, artist = self.eq.plot(name, ax=self.ax, put_labels=True)
        except Exception as exc:
            self.messagebox.showerror("Plot failed", f"Could not plot '{name}':\n{exc}")
            return

        if is_2d:
            self._current_plot_is_2d = True
            if hasattr(artist, "colorbar") and artist.colorbar is not None:
                self._active_colorbar = artist.colorbar
            else:
                self._active_colorbar = self.figure.colorbar(artist, ax=self.ax)
            self.ax.set_aspect("equal")
        else:
            self._current_plot_is_2d = False
            self.ax.grid(True, alpha=0.35)

        # Ensure labels are always visible in the GUI, even when reusing axes.
        if var.ndim >= 1:
            x_name = list(var.coords.keys())[0]
            x_da = var.coords[x_name]
            x_label = x_da.attrs.get("short_name", x_name)
            x_units = x_da.attrs.get("units", "")
            self.ax.set_xlabel(f"{x_label} [{x_units}]" if x_units else x_label)
        if is_2d and var.ndim >= 2:
            y_name = list(var.coords.keys())[1]
            y_da = var.coords[y_name]
            y_label = y_da.attrs.get("short_name", y_name)
            y_units = y_da.attrs.get("units", "")
            self.ax.set_ylabel(f"{y_label} [{y_units}]" if y_units else y_label)
        else:
            y_label = var.attrs.get("short_name", var.name or "value")
            y_units = var.attrs.get("units", "")
            self.ax.set_ylabel(f"{y_label} [{y_units}]" if y_units else y_label)

        self.ax.set_title(name)
        self.canvas.draw_idle()
        self._set_status(f"Plotted {name}.")

    def on_clear_axis(self) -> None:
        self.ax.clear()
        self._remove_colorbar()
        self._current_plot_is_2d = False
        self.ax.set_xlabel("R [m]")
        self.ax.set_ylabel("z [m]")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_title("Cleared")
        self.canvas.draw_idle()
        self._set_status("Plot cleared.")

    def on_compute_coordinates(self) -> None:
        if self.eq is None:
            self.messagebox.showwarning("No equilibrium", "Load an equilibrium first.")
            return

        selected = self._get_selected_coordinate_systems()
        if not selected:
            self.messagebox.showwarning("No coordinate system", "Select one or more coordinate systems.")
            return

        try:
            lpsi = self._get_int_value(self.lpsi_var, "lpsi")
            ltheta = self._get_int_value(self.ltheta_var, "ltheta")
        except ValueError as exc:
            self.messagebox.showerror("Invalid input", str(exc))
            return

        failures: List[str] = []
        for coord_name in selected:
            try:
                self._set_status(f"Computing {coord_name} coordinates...")
                t0 = time.perf_counter()
                coords = self.eq.compute_coordinates(
                    coordinate_system=coord_name,
                    lpsi=lpsi,
                    ltheta=ltheta,
                )
                self._computed_coords[coord_name] = coords
                dt = time.perf_counter() - t0
                self._set_status(f"Computed {coord_name} coordinates in {dt:.2f}s.")
            except Exception as exc:
                failures.append(f"{coord_name}: {exc}")

        if failures:
            self.messagebox.showerror(
                "Coordinate computation failed",
                "Some coordinate systems failed:\n\n" + "\n".join(failures),
            )
        else:
            self._set_status(f"Computed coordinates: {', '.join(selected)}")

    def on_overlay_coordinates(self) -> None:
        if self.eq is None:
            self.messagebox.showwarning("No equilibrium", "Load an equilibrium first.")
            return

        if not self._current_plot_is_2d:
            self.messagebox.showwarning(
                "2D plot required",
                "Create a 2D plot first, then overlay magnetic coordinates.",
            )
            return

        selected = self._get_selected_coordinate_systems()
        if not selected:
            self.messagebox.showwarning("No coordinate system", "Select one or more coordinate systems.")
            return

        try:
            n_surf = self._get_int_value(self.nsurf_var, "#surfaces")
        except ValueError as exc:
            self.messagebox.showerror("Invalid input", str(exc))
            return

        # Auto-compute any missing systems using current GUI settings.
        try:
            lpsi = self._get_int_value(self.lpsi_var, "lpsi")
            ltheta = self._get_int_value(self.ltheta_var, "ltheta")
        except ValueError as exc:
            self.messagebox.showerror("Invalid input", str(exc))
            return

        cmap = self.figure.colormaps.get_cmap("tab10")

        for i, coord_name in enumerate(selected):
            if coord_name not in self._computed_coords:
                try:
                    self._set_status(f"Computing missing {coord_name} coordinates...")
                    self._computed_coords[coord_name] = self.eq.compute_coordinates(
                        coordinate_system=coord_name,
                        lpsi=lpsi,
                        ltheta=ltheta,
                    )
                except Exception as exc:
                    self.messagebox.showerror(
                        "Overlay failed",
                        f"Could not compute '{coord_name}' for overlay:\n{exc}",
                    )
                    return

            coords = self._computed_coords[coord_name]
            r_inv = np.asarray(coords.coords.R_inv.values, dtype=float)
            z_inv = np.asarray(coords.coords.z_inv.values, dtype=float)
            if r_inv.ndim != 2 or z_inv.ndim != 2 or r_inv.shape != z_inv.shape:
                continue

            color = cmap(i % 10)
            psi_idx = sample_indices(r_inv.shape[0], n_surf)
            theta_idx = sample_indices(r_inv.shape[1], max(4, min(10, n_surf)))

            for j, idx in enumerate(psi_idx):
                label = coord_name if j == 0 else None
                self.ax.plot(r_inv[idx, :], z_inv[idx, :], color=color, alpha=0.85, linewidth=0.8, label=label)

            # Draw a few orthogonal lines to visualize the grid topology.
            for idx in theta_idx:
                self.ax.plot(r_inv[:, idx], z_inv[:, idx], color=color, alpha=0.35, linewidth=0.6)

        self.ax.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()
        self._set_status("Magnetic-coordinate overlays added.")


def _import_gui_modules():
    """
    Import GUI modules lazily so importing pycocos itself does not require Tk.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise RuntimeError(
            "Tkinter is required for pycocos GUI but is not available in this Python environment."
        ) from exc

    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
    except Exception as exc:
        raise RuntimeError(
            "matplotlib with Tk backend is required for pycocos GUI plotting."
        ) from exc

    return tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk


def main() -> None:
    """
    Launch the pycocos graphical viewer.
    """
    tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk = (
        _import_gui_modules()
    )

    root = tk.Tk()
    app = EquilibriumGuiApp(
        root=root,
        tk_mod=tk,
        ttk_mod=ttk,
        filedialog_mod=filedialog,
        messagebox_mod=messagebox,
        FigureCls=Figure,
        FigureCanvasTkAggCls=FigureCanvasTkAgg,
        NavigationToolbar2TkCls=NavigationToolbar2Tk,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
