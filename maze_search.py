import tkinter as tk
from tkinter import messagebox, simpledialog
import heapq
import random

# ─── Constants ────────────────────────────────────────────────

ROWS, COLS = 18, 28
CELL = 25
WIDTH  = COLS * CELL
HEIGHT = ROWS * CELL

EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
EXPLORED = 4
PATH     = 5

COLORS = {
    EMPTY:    "#F5F5F0",
    WALL:     "#2C2C2A",
    START:    "#1D9E75",
    GOAL:     "#D85A30",
    EXPLORED: "#AED6F1",
    PATH:     "#F9CA74",
}

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ─── Helpers 

def _valid(grid, r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and grid[r][c] != WALL

def _trace(parent, goal_key):
    path, cur = [], goal_key
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


#  Search Algorithms 

def bfs(grid, start, goal_set, heuristic=None):
    """BFS — guaranteed shortest path to nearest reachable goal."""
    queue = [start]
    parent = {start: None}
    visited = {start}
    order = []

    while queue:
        cur = queue.pop(0)
        order.append(cur)
        if cur in goal_set:
            return order, _trace(parent, cur), cur
        r, c = cur
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if _valid(grid, *nb) and nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)

    return order, [], None


def dfs(grid, start, goal_set, heuristic=None):
    """DFS — mark visited when popped to keep parent tracking correct."""
    stack = [(start, None)]
    visited = set()
    parent = {}
    order = []

    while stack:
        cur, par = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        parent[cur] = par
        order.append(cur)

        if cur in goal_set:
            return order, _trace(parent, cur), cur

        r, c = cur
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if _valid(grid, *nb) and nb not in visited:
                stack.append((nb, cur))

    return order, [], None


def astar(grid, start, goal_set, heuristic=None):
    """A* — uses manual heuristic if provided, else Manhattan distance."""
    goal_coords = list(goal_set)

    def h(cell):
        if heuristic and cell in heuristic:
            return heuristic[cell]
        return min(abs(cell[0] - g[0]) + abs(cell[1] - g[1]) for g in goal_coords)

    open_set = [(h(start), 0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    visited = set()
    order = []

    while open_set:
        f, g, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        order.append(cur)

        if cur in goal_set:
            return order, _trace(parent, cur), cur

        r, c = cur
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if not _valid(grid, *nb) or nb in visited:
                continue
            ng = g + 1
            if ng < g_cost.get(nb, float('inf')):
                g_cost[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_set, (ng + h(nb), ng, nb))

    return order, [], None


# ─── GUI  ───

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze — Multi-Goal AI Search")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0EFEA")

        self.grid = self._mk_grid()
        self.start_cell = None
        self.goals = []
        self.final_path = []
        self.heuristic = {}   

        self.draw_mode = tk.StringVar(value="wall")
        self.algo_var  = tk.StringVar(value="astar")
        self.speed_var = tk.IntVar(value=6)
        self.running   = False
        self._after_id = None

        self.stat_explored = tk.IntVar(value=0)
        self.stat_path     = tk.IntVar(value=0)
        self.stat_goal     = tk.StringVar(value="—")
        self.status_text   = tk.StringVar(value="Ready — set start & goals")

        self._build_ui()
        self._render()

    #  helpers 

    def _mk_grid(self):
        return [[EMPTY] * COLS for _ in range(ROWS)]

    def _get_delay(self):
        return max(4, 200 - self.speed_var.get() * 18)

    #  UI  

    def _build_ui(self):
        left = tk.Frame(self.root, bg="#F0EFEA", padx=10, pady=10)
        left.grid(row=0, column=0, sticky="ns")

        tk.Label(left, text="Maze — Multi-Goal Search",
                 font=("Helvetica", 13, "bold"), bg="#F0EFEA").pack(anchor="w", pady=(0, 10))

        self._section(left, "Draw Mode")
        for text, val in [("Wall", "wall"), ("Start", "start"),
                          ("Goal (+)", "goal"), ("Heuristic (h)", "heuristic"), ("Erase", "erase")]:
            rb = tk.Radiobutton(left, text=text, variable=self.draw_mode, value=val,
                           bg="#F0EFEA", font=("Helvetica", 9))
            rb.pack(anchor="w")

        # Heuristic info label
        self.h_info = tk.Label(left, text="Click cell to set h(n)",
                               bg="#F0EFEA", fg="#888", font=("Helvetica", 8, "italic"),
                               wraplength=155, justify="left")
        self.h_info.pack(anchor="w", padx=(16, 0))

        tk.Button(left, text="✕ Clear Heuristics", command=self._clear_heuristics,
                  font=("Helvetica", 8), relief="flat", bg="#E8E7E2",
                  fg="#555", pady=2, cursor="hand2").pack(anchor="w", pady=(2, 0))

        self._section(left, "Algorithm")
        for text, val in [("A*  (informed)",   "astar"),
                          ("BFS (uninformed)",  "bfs"),
                          ("DFS (uninformed)",  "dfs")]:
            tk.Radiobutton(left, text=text, variable=self.algo_var, value=val,
                           bg="#F0EFEA", font=("Helvetica", 9)).pack(anchor="w")

        # A* heuristic mode note
        self.h_mode_label = tk.Label(left,
            text="A*: using Manhattan dist",
            bg="#F0EFEA", fg="#1D9E75", font=("Helvetica", 8, "italic"),
            wraplength=155, justify="left")
        self.h_mode_label.pack(anchor="w", padx=(16, 0))

        self._section(left, "Speed")
        tk.Scale(left, from_=1, to=10, orient="horizontal", variable=self.speed_var,
                 bg="#F0EFEA", length=150).pack(anchor="w")

        self._section(left, "")
        btn_cfg = dict(width=16, font=("Helvetica", 9), relief="flat",
                       bg="#2C2C2A", fg="white", activebackground="#444",
                       activeforeground="white", pady=4, cursor="hand2")
        tk.Button(left, text="▶  Run",           command=self._run,           **btn_cfg).pack(pady=2)
        tk.Button(left, text="↺  Reset paths",   command=self._reset_paths,   **btn_cfg).pack(pady=2)
        tk.Button(left, text="✕  Clear all",     command=self._clear_all,     **btn_cfg).pack(pady=2)
        tk.Button(left, text="⚡  Generate maze", command=self._generate_maze, **btn_cfg).pack(pady=2)

        self._section(left, "Stats")
        for label, var in [("Explored:", self.stat_explored),
                            ("Path len:", self.stat_path),
                            ("Goal:",     self.stat_goal)]:
            row = tk.Frame(left, bg="#F0EFEA")
            row.pack(anchor="w", fill="x")
            tk.Label(row, text=label, bg="#F0EFEA", font=("Helvetica", 9),
                     width=10, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, bg="#F0EFEA",
                     font=("Helvetica", 9, "bold")).pack(side="left")

        tk.Label(left, text="", bg="#F0EFEA").pack()
        tk.Label(left, textvariable=self.status_text, bg="#F0EFEA",
                 font=("Helvetica", 9, "bold"), fg="#1D9E75",
                 wraplength=155, justify="left").pack(anchor="w")

        self._section(left, "Legend")
        for name, color in [("Start",      COLORS[START]),
                             ("Goal",       COLORS[GOAL]),
                             ("Wall",       COLORS[WALL]),
                             ("Explored",   COLORS[EXPLORED]),
                             ("Path",       COLORS[PATH]),
                             ("Path line",  "#e74c3c"),
                             ("Heuristic",  "#C39BD3")]:
            row = tk.Frame(left, bg="#F0EFEA")
            row.pack(anchor="w", pady=1)
            tk.Label(row, bg=color, width=2, relief="flat").pack(side="left", padx=(0, 6))
            tk.Label(row, text=name, bg="#F0EFEA", font=("Helvetica", 9)).pack(side="left")

        # Canvas
        right = tk.Frame(self.root, bg="#E8E7E2", padx=6, pady=10)
        right.grid(row=0, column=1, sticky="nsew")

        self.canvas = tk.Canvas(right, width=WIDTH, height=HEIGHT,
                                bg=COLORS[EMPTY], highlightthickness=1,
                                highlightbackground="#CCCCCC")
        self.canvas.pack()
        self.canvas.bind("<Button-1>",  self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Watch algo selection to update hint label
        self.algo_var.trace_add("write", self._update_h_mode_label)
        self.draw_mode.trace_add("write", self._update_h_mode_label)

    def _update_h_mode_label(self, *_):
        algo = self.algo_var.get()
        if algo != "astar":
            self.h_mode_label.config(text="(Heuristic used by A* only)", fg="#aaa")
            return
        if self.heuristic:
            count = len(self.heuristic)
            self.h_mode_label.config(
                text=f"A*: {count} manual h(n) set\n(others use Manhattan)",
                fg="#D85A30")
        else:
            self.h_mode_label.config(text="A*: using Manhattan dist", fg="#1D9E75")

    def _section(self, parent, title):
        tk.Label(parent, text="", bg="#F0EFEA").pack()
        if title:
            tk.Label(parent, text=title, font=("Helvetica", 10, "bold"),
                     bg="#F0EFEA").pack(anchor="w")

    # Canvas 

    def _render(self):
        self.canvas.delete("all")

        # 1) cells
        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = c * CELL + 1, r * CELL + 1
                x1, y1 = x0 + CELL - 2, y0 + CELL - 2
                cell_val = self.grid[r][c]

                # heuristic cells get a purple tint (if empty/explored/path)
                if (r, c) in self.heuristic and cell_val in (EMPTY, EXPLORED, PATH):
                    fill = "#C39BD3"
                else:
                    fill = COLORS.get(cell_val, COLORS[EMPTY])

                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=fill, outline=""
                )

        # 2) heuristic values drawn as small text
        for (r, c), hval in self.heuristic.items():
            if self.grid[r][c] not in (WALL, START, GOAL):
                self.canvas.create_text(
                    c * CELL + CELL // 2,
                    r * CELL + CELL // 2,
                    text=str(hval),
                    fill="#6C3483",
                    font=("Helvetica", 7, "bold")
                )

        # 3) red path line drawn on top of cells
        if len(self.final_path) > 1:
            coords = []
            for r, c in self.final_path:
                coords += [c * CELL + CELL // 2, r * CELL + CELL // 2]
            self.canvas.create_line(
                *coords,
                fill="#e74c3c", width=3,
                capstyle="round", joinstyle="round"
            )
            r1, c1 = self.final_path[-2]
            r2, c2 = self.final_path[-1]
            self.canvas.create_line(
                c1 * CELL + CELL // 2, r1 * CELL + CELL // 2,
                c2 * CELL + CELL // 2, r2 * CELL + CELL // 2,
                fill="#e74c3c", width=3,
                arrow="last", arrowshape=(10, 12, 4)
            )

        # 4) labels (S and G1, G2, …)
        font = ("Helvetica", 8, "bold")
        if self.start_cell:
            r, c = self.start_cell
            self.canvas.create_text(
                c * CELL + CELL // 2, r * CELL + CELL // 2,
                text="S", fill="white", font=font
            )
        for i, (r, c) in enumerate(self.goals):
            self.canvas.create_text(
                c * CELL + CELL // 2, r * CELL + CELL // 2,
                text=f"G{i+1}", fill="white", font=font
            )

        # 5) grid lines
        for r in range(ROWS + 1):
            self.canvas.create_line(0, r * CELL, WIDTH, r * CELL,
                                    fill="#DDDDD8", width=0.5)
        for c in range(COLS + 1):
            self.canvas.create_line(c * CELL, 0, c * CELL, HEIGHT,
                                    fill="#DDDDD8", width=0.5)

    def _cell_at(self, event):
        c = event.x // CELL
        r = event.y // CELL
        if 0 <= r < ROWS and 0 <= c < COLS:
            return r, c
        return None

    def _set_cell(self, r, c):
        mode = self.draw_mode.get()
        if mode == "wall":
            if self.start_cell == (r, c):
                self.start_cell = None
            self.goals = [g for g in self.goals if g != (r, c)]
            self.heuristic.pop((r, c), None)
            self.grid[r][c] = WALL
        elif mode == "erase":
            if self.start_cell == (r, c):
                self.start_cell = None
            self.goals = [g for g in self.goals if g != (r, c)]
            self.heuristic.pop((r, c), None)
            self.grid[r][c] = EMPTY
        elif mode == "start":
            if self.start_cell:
                self.grid[self.start_cell[0]][self.start_cell[1]] = EMPTY
            self.start_cell = (r, c)
            self.grid[r][c] = START
        elif mode == "goal":
            if (r, c) in self.goals:
                self.goals.remove((r, c))
                self.grid[r][c] = EMPTY
            else:
                self.goals.append((r, c))
                self.grid[r][c] = GOAL
        elif mode == "heuristic":
            self._ask_heuristic(r, c)
            return   
        self._render()

    def _ask_heuristic(self, r, c):
        """Prompt user for a heuristic value for cell (r,c)."""
        if self.grid[r][c] == WALL:
            messagebox.showwarning("Invalid", "Cannot set heuristic on a wall cell.")
            return
        current = self.heuristic.get((r, c), "")
        val = simpledialog.askstring(
            "Set Heuristic h(n)",
            f"Enter h(n) value for cell ({r},{c}):\n"
            f"(leave empty or 0 to remove)",
            initialvalue=str(current),
            parent=self.root
        )
        if val is None:
            return
        val = val.strip()
        if val == "" or val == "0":
            self.heuristic.pop((r, c), None)
        else:
            try:
                self.heuristic[(r, c)] = int(val)
            except ValueError:
                messagebox.showerror("Invalid", "Please enter an integer value.")
                return
        self._update_h_mode_label()
        self._render()

    def _on_click(self, event):
        if self.running:
            return
        cell = self._cell_at(event)
        if cell:
            self._set_cell(*cell)

    def _on_drag(self, event):
        if self.running:
            return
        if self.draw_mode.get() == "heuristic":
            return
        cell = self._cell_at(event)
        if cell:
            self._set_cell(*cell)

    # ─ Run  

    def _run(self):
        if self.running:
            return
        if not self.start_cell:
            messagebox.showwarning("Missing", "Set a start cell first!")
            return
        if not self.goals:
            messagebox.showwarning("Missing", "Add at least one goal!")
            return

        self._reset_paths(silent=True)
        self.running = True
        self.final_path = []
        self.status_text.set("Searching…")

        goal_set = set(self.goals)
        algo = self.algo_var.get()
        fn = {"astar": astar, "bfs": bfs, "dfs": dfs}[algo]

        # Pass heuristic dict to A* (ignored by BFS/DFS)
        h_arg = self.heuristic if (algo == "astar" and self.heuristic) else None
        order, path, goal_key = fn(self.grid, self.start_cell, goal_set, heuristic=h_arg)

        delay = self._get_delay()
        inner = [cell for cell in order
                 if cell != self.start_cell and cell not in self.goals]

        exp_count = [0]

        def step_explore(i=0):
            if i < len(inner):
                r, c = inner[i]
                if self.grid[r][c] == EMPTY:
                    self.grid[r][c] = EXPLORED
                    exp_count[0] += 1
                self.stat_explored.set(exp_count[0])
                self._render()
                self._after_id = self.root.after(delay, step_explore, i + 1)
            else:
                if not path:
                    self.status_text.set("No path found!")
                    self.running = False
                    return
                goal_idx = self.goals.index(goal_key)
                self.stat_goal.set(f"G{goal_idx + 1}")
                path_inner = [cell for cell in path if cell != self.start_cell]
                step_path(path_inner, 0)

        def step_path(path_inner, j=0):
            if j < len(path_inner):
                r, c = path_inner[j]
                if self.grid[r][c] != GOAL:
                    self.grid[r][c] = PATH
                self.stat_path.set(len(path) - 1)
                self.final_path = path[:j + 2]
                self._render()
                self._after_id = self.root.after(delay // 2, step_path, path_inner, j + 1)
            else:
                self.final_path = list(path)
                self._render()
                goal_idx = self.goals.index(goal_key)
                h_note = " [manual h]" if h_arg else ""
                self.status_text.set(
                    f"{algo.upper()}{h_note}: G{goal_idx + 1} — "
                    f"{len(path) - 1} steps, explored {exp_count[0]} cells"
                )
                self.running = False

        step_explore()

    #  Controls 

    def _reset_paths(self, silent=False):
        if self.running and not silent:
            return
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.running = False
        self.final_path = []
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] in (EXPLORED, PATH):
                    self.grid[r][c] = EMPTY
        self.stat_explored.set(0)
        self.stat_path.set(0)
        self.stat_goal.set("—")
        self.status_text.set("Ready")
        self._render()

    def _clear_heuristics(self):
        self.heuristic.clear()
        self._update_h_mode_label()
        self._render()

    def _clear_all(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.running = False
        self.grid = self._mk_grid()
        self.start_cell = None
        self.goals = []
        self.final_path = []
        self.heuristic.clear()
        self.stat_explored.set(0)
        self.stat_path.set(0)
        self.stat_goal.set("—")
        self.status_text.set("Ready — set start & goals")
        self._update_h_mode_label()
        self._render()

    def _generate_maze(self):
        """Recursive backtracker maze generation."""
        self._clear_all()
        for r in range(ROWS):
            for c in range(COLS):
                self.grid[r][c] = WALL

        def carve(r, c):
            dirs = list(DIRS)
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr * 2, c + dc * 2
                if 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] == WALL:
                    self.grid[r + dr][c + dc] = EMPTY
                    self.grid[nr][nc] = EMPTY
                    carve(nr, nc)

        self.grid[1][1] = EMPTY
        carve(1, 1)

        self.start_cell = (1, 1)
        self.grid[1][1] = START

        goal_positions = [
            (1, COLS - 2),
            (ROWS - 2, 1),
            (ROWS - 2, COLS - 2),
            (ROWS // 2, COLS // 2),
        ]
        self.goals = []
        for r, c in goal_positions:
            self.grid[r][c] = GOAL
            self.goals.append((r, c))

        self.status_text.set("Maze ready — press Run!")
        self._render()


# ─── Main  ──

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
