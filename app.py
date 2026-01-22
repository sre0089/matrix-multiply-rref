from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Optional, Union
from fractions import Fraction

MatrixF = List[List[float]]
MatrixX = List[List[Union[float, Fraction]]]


# =========================
# Optional NumPy acceleration
# =========================
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore


# =========================
# Core helpers
# =========================
def shape(A: List[List[object]]) -> Tuple[int, int]:
    if not A or not isinstance(A, list):
        raise ValueError("Matrix must be a non-empty list of rows.")
    m = len(A)
    n = len(A[0])
    if n == 0:
        raise ValueError("Matrix must have at least 1 column.")
    for r in A:
        if len(r) != n:
            raise ValueError("All rows must have the same number of columns.")
    return m, n


def max_abs(A: MatrixF) -> float:
    mx = 0.0
    for row in A:
        for v in row:
            av = abs(v)
            if av > mx:
                mx = av
    return mx


# =========================
# Parsing / formatting
# =========================
def parse_matrix(text: str) -> MatrixF:
    """
    Fast, forgiving parser:
      - rows separated by newlines
      - values separated by spaces and/or commas
    Example:
      1 2 3
      4,5,6
    """
    raw = text.strip()
    if not raw:
        raise ValueError("Matrix input is empty.")

    rows: MatrixF = []
    expected_cols: Optional[int] = None

    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = [p for p in line.replace(",", " ").split() if p]
        if not parts:
            continue

        try:
            row = [float(x) for x in parts]
        except ValueError as e:
            raise ValueError(f"Row {i}: could not parse a number. ({e})")

        if expected_cols is None:
            expected_cols = len(row)
            if expected_cols == 0:
                raise ValueError("Matrix must have at least 1 column.")
        elif len(row) != expected_cols:
            raise ValueError(
                f"Row {i}: has {len(row)} entries, expected {expected_cols}. "
                f"(Check missing/extra numbers on that line.)"
            )

        rows.append(row)

    if not rows:
        raise ValueError("No rows found.")
    return rows


def format_matrix(A: MatrixX, decimals: int = 4, mode: str = "float") -> str:
    if mode == "fraction":
        def f(x: Union[float, Fraction]) -> str:
            if isinstance(x, Fraction):
                return str(x)
            return str(Fraction(x).limit_denominator())
        return "\n".join(" ".join(f(v) for v in row) for row in A)

    fmt = f"{{:.{decimals}f}}"
    def f(x: Union[float, Fraction]) -> str:
        if isinstance(x, Fraction):
            # fallback: render fraction as float if it sneaks in
            return fmt.format(float(x))
        return fmt.format(x)
    return "\n".join(" ".join(f(v) for v in row) for row in A)


# =========================
# Multiply (Python + NumPy)
# =========================
def matmul_python(A: MatrixF, B: MatrixF) -> MatrixF:
    m, n = shape(A)
    n2, p = shape(B)
    if n != n2:
        raise ValueError(f"Incompatible shapes: A is {m}x{n}, B is {n2}x{p}.")
    C: MatrixF = [[0.0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
    return C


def matmul_numpy(A: MatrixF, B: MatrixF) -> MatrixF:
    assert HAS_NUMPY and np is not None
    a = np.array(A, dtype=float)
    b = np.array(B, dtype=float)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: A is {a.shape[0]}x{a.shape[1]}, B is {b.shape[0]}x{b.shape[1]}.")
    c = a @ b
    return c.tolist()


# =========================
# RREF (Float: Python + NumPy, Fraction: exact)
# =========================
def rref_float_python(A: MatrixF, eps: float = 1e-12) -> Tuple[MatrixF, List[int]]:
    R = [row[:] for row in A]
    m, n = shape(R)

    # scale-aware eps
    scale = max(1.0, max_abs(R))
    eps_eff = eps * scale

    pivots: List[int] = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot_row: Optional[int] = None
        best = eps_eff
        for r in range(row, m):
            v = abs(R[r][col])
            if v > best:
                best = v
                pivot_row = r
        if pivot_row is None:
            continue

        if pivot_row != row:
            R[row], R[pivot_row] = R[pivot_row], R[row]

        pivot = R[row][col]
        for j in range(col, n):
            R[row][j] /= pivot

        for r in range(m):
            if r == row:
                continue
            factor = R[r][col]
            if abs(factor) <= eps_eff:
                continue
            for j in range(col, n):
                R[r][j] -= factor * R[row][j]

        # clean tiny noise
        for r in range(m):
            for j in range(n):
                if abs(R[r][j]) < eps_eff:
                    R[r][j] = 0.0

        pivots.append(col)
        row += 1

    return R, pivots


def rref_float_numpy(A: MatrixF, eps: float = 1e-12) -> Tuple[MatrixF, List[int]]:
    assert HAS_NUMPY and np is not None
    R = np.array(A, dtype=float)
    m, n = R.shape

    scale = max(1.0, float(np.max(np.abs(R))) if R.size else 1.0)
    eps_eff = eps * scale

    pivots: List[int] = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        # pivot selection (partial pivoting)
        col_slice = np.abs(R[row:, col])
        idx = int(np.argmax(col_slice))
        pivot_row = row + idx
        if abs(R[pivot_row, col]) <= eps_eff:
            continue

        if pivot_row != row:
            R[[row, pivot_row], :] = R[[pivot_row, row], :]

        pivot = R[row, col]
        R[row, col:] = R[row, col:] / pivot

        # eliminate all other rows
        factors = R[:, col].copy()
        factors[row] = 0.0
        # R[r, col:] -= factors[r] * R[row, col:]
        R[:, col:] = R[:, col:] - factors[:, None] * R[row, col:]

        # clean tiny noise
        R[np.abs(R) < eps_eff] = 0.0

        pivots.append(col)
        row += 1

    return R.tolist(), pivots


def rref_fraction(A: MatrixF) -> Tuple[List[List[Fraction]], List[int]]:
    # Convert to Fractions (exact)
    R: List[List[Fraction]] = [[Fraction(x).limit_denominator() for x in row] for row in A]
    m, n = shape(R)

    pivots: List[int] = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        # find any non-zero pivot (exact)
        pivot_row = None
        for r in range(row, m):
            if R[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        if pivot_row != row:
            R[row], R[pivot_row] = R[pivot_row], R[row]

        pivot = R[row][col]
        # normalize
        for j in range(col, n):
            R[row][j] /= pivot

        # eliminate all other rows
        for r in range(m):
            if r == row:
                continue
            factor = R[r][col]
            if factor == 0:
                continue
            for j in range(col, n):
                R[r][j] -= factor * R[row][j]

        pivots.append(col)
        row += 1

    return R, pivots


# =========================
# GUI
# =========================
class MatrixToolApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master = master
        self.pack(fill="both", expand=True)

        self.master.title("Matrix Tool — Multiply + RREF")
        self.master.minsize(980, 620)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Monospace for matrices (makes alignment feel way better)
        self.mono_font = ("Courier New", 11)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.tab_mul = ttk.Frame(self.nb, padding=10)
        self.tab_rref = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.tab_mul, text="Multiply (A × B)")
        self.nb.add(self.tab_rref, text="Row Reduce (RREF)")

        # status bar
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(8, 0))

        self._build_multiply_tab()
        self._build_rref_tab()

    # ---------- UI helpers ----------
    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _show_error(self, title: str, msg: str):
        self._set_status("Error.")
        messagebox.showerror(title, msg)

    def _shape_str(self, txt: str) -> str:
        try:
            A = parse_matrix(txt)
            m, n = shape(A)
            return f"{m}×{n}"
        except Exception:
            return "—"

    def _make_text_area(self, parent, height=12):
        txt = tk.Text(parent, height=height, wrap="none", undo=True, font=self.mono_font)
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=txt.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        return txt, yscroll, xscroll

    def _copy_to_clipboard(self, content: str):
        self.master.clipboard_clear()
        self.master.clipboard_append(content)
        self._set_status("Copied to clipboard.")

    # ---------- Multiply tab ----------
    def _build_multiply_tab(self):
        self.tab_mul.columnconfigure(0, weight=1)
        self.tab_mul.columnconfigure(1, weight=1)
        self.tab_mul.columnconfigure(2, weight=1)
        self.tab_mul.rowconfigure(2, weight=1)

        # Header row: label + shape
        self.mul_shapeA = tk.StringVar(value="Shape: —")
        self.mul_shapeB = tk.StringVar(value="Shape: —")
        self.mul_shapeC = tk.StringVar(value="Shape: —")

        hdrA = ttk.Frame(self.tab_mul)
        hdrA.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(hdrA, text="Matrix A").pack(side="left")
        ttk.Label(hdrA, textvariable=self.mul_shapeA).pack(side="right")

        hdrB = ttk.Frame(self.tab_mul)
        hdrB.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(hdrB, text="Matrix B").pack(side="left")
        ttk.Label(hdrB, textvariable=self.mul_shapeB).pack(side="right")

        hdrC = ttk.Frame(self.tab_mul)
        hdrC.grid(row=0, column=2, sticky="ew")
        ttk.Label(hdrC, text="Output: A × B").pack(side="left")
        ttk.Label(hdrC, textvariable=self.mul_shapeC).pack(side="right")

        # Text areas row
        self.mul_A, yA, xA = self._make_text_area(self.tab_mul, height=14)
        self.mul_B, yB, xB = self._make_text_area(self.tab_mul, height=14)
        self.mul_out, yC, xC = self._make_text_area(self.tab_mul, height=14)
        self.mul_out.configure(state="disabled")

        self.mul_A.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        yA.grid(row=2, column=0, sticky="nse", padx=(0, 8))
        xA.grid(row=3, column=0, sticky="ew", padx=(0, 8))

        self.mul_B.grid(row=2, column=1, sticky="nsew", padx=(0, 8))
        yB.grid(row=2, column=1, sticky="nse", padx=(0, 8))
        xB.grid(row=3, column=1, sticky="ew", padx=(0, 8))

        self.mul_out.grid(row=2, column=2, sticky="nsew")
        yC.grid(row=2, column=2, sticky="nse")
        xC.grid(row=3, column=2, sticky="ew")

        # Controls
        controls = ttk.Frame(self.tab_mul)
        controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        controls.columnconfigure(12, weight=1)

        ttk.Button(controls, text="Compute (Ctrl/⌘+Enter)", command=self.on_mul_compute).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Clear", command=self.on_mul_clear).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(controls, text="Load Example", command=self.on_mul_example).grid(row=0, column=2, padx=(0, 8))

        ttk.Button(controls, text="Copy Output", command=self.on_mul_copy).grid(row=0, column=3, padx=(16, 8))

        ttk.Label(controls, text="Decimals:").grid(row=0, column=4, padx=(16, 6))
        self.mul_decimals = tk.IntVar(value=4)
        ttk.Spinbox(controls, from_=0, to=10, width=5, textvariable=self.mul_decimals).grid(row=0, column=5)

        self.use_numpy_mul = tk.BooleanVar(value=HAS_NUMPY)
        numpy_label = "Use NumPy (fast)" if HAS_NUMPY else "Use NumPy (not installed)"
        chk = ttk.Checkbutton(controls, text=numpy_label, variable=self.use_numpy_mul)
        if not HAS_NUMPY:
            chk.state(["disabled"])
            self.use_numpy_mul.set(False)
        chk.grid(row=0, column=6, padx=(16, 0), sticky="w")

        # bindings
        self.master.bind_all("<Control-Return>", lambda e: self._compute_current_tab())
        self.master.bind_all("<Command-Return>", lambda e: self._compute_current_tab())

        # live shape preview
        self.mul_A.bind("<KeyRelease>", lambda e: self._update_mul_shapes())
        self.mul_B.bind("<KeyRelease>", lambda e: self._update_mul_shapes())
        self.on_mul_example()
        self._update_mul_shapes()

    def _update_mul_shapes(self):
        self.mul_shapeA.set(f"Shape: {self._shape_str(self.mul_A.get('1.0','end'))}")
        self.mul_shapeB.set(f"Shape: {self._shape_str(self.mul_B.get('1.0','end'))}")
        # output shape computed after compute
        if self.mul_out.get("1.0", "end").strip():
            # best effort
            self.mul_shapeC.set(f"Shape: {self._shape_str(self.mul_out.get('1.0','end'))}")
        else:
            self.mul_shapeC.set("Shape: —")

    def on_mul_compute(self):
        try:
            A = parse_matrix(self.mul_A.get("1.0", "end"))
            B = parse_matrix(self.mul_B.get("1.0", "end"))

            if self.use_numpy_mul.get() and HAS_NUMPY:
                C = matmul_numpy(A, B)
                backend = "NumPy"
            else:
                C = matmul_python(A, B)
                backend = "Python"

            out = format_matrix(C, decimals=int(self.mul_decimals.get()), mode="float")

            self.mul_out.configure(state="normal")
            self.mul_out.delete("1.0", "end")
            self.mul_out.insert("1.0", out)
            self.mul_out.configure(state="disabled")

            m, n = shape(A)
            _, p = shape(C)
            self.mul_shapeC.set(f"Shape: {m}×{p}")
            self._set_status(f"Computed A×B using {backend}: ({m}x{n}) -> ({m}x{p}).")
        except Exception as e:
            self._show_error("Multiply Error", str(e))
        finally:
            self._update_mul_shapes()

    def on_mul_copy(self):
        content = self.mul_out.get("1.0", "end").strip()
        if not content:
            self._set_status("Nothing to copy.")
            return
        self._copy_to_clipboard(content)

    def on_mul_clear(self):
        self.mul_A.delete("1.0", "end")
        self.mul_B.delete("1.0", "end")
        self.mul_out.configure(state="normal")
        self.mul_out.delete("1.0", "end")
        self.mul_out.configure(state="disabled")
        self.mul_shapeC.set("Shape: —")
        self._update_mul_shapes()
        self._set_status("Cleared multiply inputs.")

    def on_mul_example(self):
        self.mul_A.delete("1.0", "end")
        self.mul_B.delete("1.0", "end")
        self.mul_A.insert("1.0", "1 2 3\n4 5 6")
        self.mul_B.insert("1.0", "7 8\n9 10\n11 12")
        self._set_status("Loaded multiply example.")
        self._update_mul_shapes()

    # ---------- RREF tab ----------
    def _build_rref_tab(self):
        self.tab_rref.columnconfigure(0, weight=1)
        self.tab_rref.columnconfigure(1, weight=1)
        self.tab_rref.rowconfigure(2, weight=1)

        self.rref_shape_in = tk.StringVar(value="Shape: —")
        self.rref_shape_out = tk.StringVar(value="Shape: —")

        hdrL = ttk.Frame(self.tab_rref)
        hdrL.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(hdrL, text="Matrix (or augmented matrix)").pack(side="left")
        ttk.Label(hdrL, textvariable=self.rref_shape_in).pack(side="right")

        hdrR = ttk.Frame(self.tab_rref)
        hdrR.grid(row=0, column=1, sticky="ew")
        ttk.Label(hdrR, text="Output: RREF").pack(side="left")
        ttk.Label(hdrR, textvariable=self.rref_shape_out).pack(side="right")

        self.rref_in, yM, xM = self._make_text_area(self.tab_rref, height=16)
        self.rref_out, yR, xR = self._make_text_area(self.tab_rref, height=16)
        self.rref_out.configure(state="disabled")

        self.rref_in.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        yM.grid(row=2, column=0, sticky="nse", padx=(0, 8))
        xM.grid(row=3, column=0, sticky="ew", padx=(0, 8))

        self.rref_out.grid(row=2, column=1, sticky="nsew")
        yR.grid(row=2, column=1, sticky="nse")
        xR.grid(row=3, column=1, sticky="ew")

        controls = ttk.Frame(self.tab_rref)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        controls.columnconfigure(20, weight=1)

        ttk.Button(controls, text="Compute RREF (Ctrl/⌘+Enter)", command=self.on_rref_compute).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Clear", command=self.on_rref_clear).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(controls, text="Load Example", command=self.on_rref_example).grid(row=0, column=2, padx=(0, 8))

        ttk.Button(controls, text="Copy Output", command=self.on_rref_copy).grid(row=0, column=3, padx=(16, 8))

        ttk.Label(controls, text="Mode:").grid(row=0, column=4, padx=(16, 6))
        self.rref_mode = tk.StringVar(value="float")
        mode = ttk.Combobox(controls, width=10, textvariable=self.rref_mode, state="readonly",
                            values=["float", "fraction"])
        mode.grid(row=0, column=5)
        mode.bind("<<ComboboxSelected>>", lambda e: self._on_rref_mode_change())

        ttk.Label(controls, text="Decimals:").grid(row=0, column=6, padx=(16, 6))
        self.rref_decimals = tk.IntVar(value=4)
        self.dec_spin = ttk.Spinbox(controls, from_=0, to=10, width=5, textvariable=self.rref_decimals)
        self.dec_spin.grid(row=0, column=7)

        ttk.Label(controls, text="Epsilon:").grid(row=0, column=8, padx=(16, 6))
        self.rref_eps = tk.DoubleVar(value=1e-12)
        self.eps_entry = ttk.Entry(controls, width=10, textvariable=self.rref_eps)
        self.eps_entry.grid(row=0, column=9)

        self.use_numpy_rref = tk.BooleanVar(value=HAS_NUMPY)
        numpy_label = "Use NumPy (fast)" if HAS_NUMPY else "Use NumPy (not installed)"
        chk = ttk.Checkbutton(controls, text=numpy_label, variable=self.use_numpy_rref)
        if not HAS_NUMPY:
            chk.state(["disabled"])
            self.use_numpy_rref.set(False)
        chk.grid(row=0, column=10, padx=(16, 0), sticky="w")

        self.pivots_var = tk.StringVar(value="Pivots: —")
        ttk.Label(controls, textvariable=self.pivots_var).grid(row=0, column=11, padx=(16, 0), sticky="w")

        # live shape preview
        self.rref_in.bind("<KeyRelease>", lambda e: self._update_rref_shapes())
        self.on_rref_example()
        self._update_rref_shapes()
        self._on_rref_mode_change()

    def _update_rref_shapes(self):
        self.rref_shape_in.set(f"Shape: {self._shape_str(self.rref_in.get('1.0','end'))}")
        if self.rref_out.get("1.0", "end").strip():
            self.rref_shape_out.set(f"Shape: {self._shape_str(self.rref_out.get('1.0','end'))}")
        else:
            self.rref_shape_out.set("Shape: —")

    def _on_rref_mode_change(self):
        mode = self.rref_mode.get()
        if mode == "fraction":
            # fractions ignore decimals/eps and numpy toggle
            self.dec_spin.state(["disabled"])
            self.eps_entry.state(["disabled"])
            # still show checkbox but disable it
            # (fraction mode is exact python; numpy irrelevant)
            # easiest: just disable via children scan
            self.use_numpy_rref.set(False)
        else:
            self.dec_spin.state(["!disabled"])
            self.eps_entry.state(["!disabled"])
            if HAS_NUMPY:
                # keep user's choice; default already set
                pass
        self._set_status(f"RREF mode set to {mode}.")

    def on_rref_compute(self):
        try:
            M = parse_matrix(self.rref_in.get("1.0", "end"))
            mode = self.rref_mode.get()

            if mode == "fraction":
                R, pivots = rref_fraction(M)
                out = format_matrix(R, mode="fraction")
                backend = "Exact (Fraction)"
            else:
                eps = float(self.rref_eps.get())
                if self.use_numpy_rref.get() and HAS_NUMPY:
                    R, pivots = rref_float_numpy(M, eps=eps)
                    backend = "NumPy"
                else:
                    R, pivots = rref_float_python(M, eps=eps)
                    backend = "Python"
                out = format_matrix(R, decimals=int(self.rref_decimals.get()), mode="float")

            self.rref_out.configure(state="normal")
            self.rref_out.delete("1.0", "end")
            self.rref_out.insert("1.0", out)
            self.rref_out.configure(state="disabled")

            self.pivots_var.set(f"Pivots: {pivots}")
            m, n = shape(M)
            self.rref_shape_out.set(f"Shape: {m}×{n}")
            self._set_status(f"Computed RREF using {backend}. Pivot cols: {pivots}.")
        except Exception as e:
            self._show_error("RREF Error", str(e))
        finally:
            self._update_rref_shapes()

    def on_rref_copy(self):
        content = self.rref_out.get("1.0", "end").strip()
        if not content:
            self._set_status("Nothing to copy.")
            return
        self._copy_to_clipboard(content)

    def on_rref_clear(self):
        self.rref_in.delete("1.0", "end")
        self.rref_out.configure(state="normal")
        self.rref_out.delete("1.0", "end")
        self.rref_out.configure(state="disabled")
        self.pivots_var.set("Pivots: —")
        self._update_rref_shapes()
        self._set_status("Cleared RREF inputs.")

    def on_rref_example(self):
        self.rref_in.delete("1.0", "end")
        self.rref_in.insert("1.0", "1 2 -1 -4\n2 3 -1 -11\n-2 0 -3 22")
        self._set_status("Loaded RREF example.")
        self._update_rref_shapes()

    # ---------- shared ----------
    def _compute_current_tab(self):
        tab = self.nb.index(self.nb.select())
        if tab == 0:
            self.on_mul_compute()
        else:
            self.on_rref_compute()


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    MatrixToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
