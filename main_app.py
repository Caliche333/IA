"""
Archivo: main_app.py (hub principal)
Descripción: Panel principal de la suite de reconocimiento facial.
Nuevo diseño: carrusel central + sidebar de módulos.
"""

import tkinter as tk
from tkinter import messagebox
import os
import sys
import subprocess

# Paleta de colores
COLOR_BG = "#020617"          # Fondo general
COLOR_SURFACE = "#020617"     # Fondo tarjetas
COLOR_SURFACE_ALT = "#0B1120" # Encabezado / barra
COLOR_ACCENT = "#4F46E5"      # Botones principales
COLOR_ACCENT_SOFT = "#22C55E" # Badges / detalles
COLOR_TEXT = "#F9FAFB"
COLOR_MUTED = "#9CA3AF"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Referencias a los scripts de cada módulo
SCRIPTS = {
    "captura": "studio_captura_dataset.py",
    "entrenar": "studio_entrenamiento_modelo.py",
    "deteccion": "studio_deteccion_tiempo_real.py",
    "diagnostico": "studio_analitica_modelo.py",
}

TOOLS = [
    {
        "key": "captura",
        "emoji": "📸",
        "title": "Captura de dataset",
        "tagline": "Registra rostros de estudiantes desde la cámara.",
        "desc": "Crea el banco de imágenes de estudiantes, organizando automáticamente las capturas por carpeta y nombre.",
        "steps": [
            "Escribe el nombre del estudiante.",
            "Añádelo a la lista.",
            "Inicia la captura y sigue las indicaciones en cámara.",
        ],
    },
    {
        "key": "entrenar",
        "emoji": "🧪",
        "title": "Entrenamiento del modelo",
        "tagline": "Ajusta epochs, batch size y learning rate.",
        "desc": "Entrena una CNN con el dataset actual, usando validación automática y guardando el modelo entrenado.",
        "steps": [
            "Verifica que el dataset esté organizado por carpetas.",
            "Ajusta epochs, batch size y LR.",
            "Lanza el entrenamiento y espera al resumen final.",
        ],
    },
    {
        "key": "deteccion",
        "emoji": "🛰️",
        "title": "Detección en vivo",
        "tagline": "Reconoce estudiantes en tiempo real con la cámara.",
        "desc": "Abre la cámara, detecta rostros y muestra el nombre y la confianza actual del modelo para cada detección.",
        "steps": [
            "Verifica que el modelo haya sido entrenado.",
            "Activa la detección desde la interfaz.",
            "Observa el nombre, confianza y FPS en vivo.",
        ],
    },
    {
        "key": "diagnostico",
        "emoji": "📊",
        "title": "Diagnóstico del modelo",
        "tagline": "Analiza métricas, matriz de confusión y reporte.",
        "desc": "Evalúa el rendimiento del modelo con accuracy, matriz de confusión y reporte detallado por clase.",
        "steps": [
            "Carga el modelo entrenado.",
            "Ejecuta el análisis sobre el dataset.",
            "Explora matriz de confusión y reporte detallado.",
        ],
    },
]


class MainApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("FaceLab Suite · Centro de control")
        self.root.geometry("1120x640")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(960, 560)

        self.current_index = 0

        self._build_layout()
        self._render_tool()

    # ---------------- LAYOUT GENERAL ---------------- #

    def _build_layout(self) -> None:
        # CONTENEDOR PRINCIPAL
        container = tk.Frame(self.root, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # ENCABEZADO SUPERIOR
        header = tk.Frame(container, bg=COLOR_SURFACE_ALT, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_block = tk.Frame(header, bg=COLOR_SURFACE_ALT)
        title_block.pack(side="left", padx=18, pady=10)

        tk.Label(
            title_block,
            text="FaceLab Suite",
            font=("Segoe UI Semibold", 22, "bold"),
            bg=COLOR_SURFACE_ALT,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            title_block,
            text="Orquesta todo el flujo: capturas, entrenamiento, detección en vivo y diagnóstico del modelo.",
            font=("Segoe UI", 9),
            bg=COLOR_SURFACE_ALT,
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        badge = tk.Label(
            header,
            text="CNN · Reconocimiento facial estudiantil",
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_ACCENT_SOFT,
            fg="white",
            padx=10,
            pady=4,
        )
        badge.pack(side="right", pady=18, padx=18)

        # CUERPO PRINCIPAL: sidebar + carrusel central + panel info
        body = tk.Frame(container, bg=COLOR_BG)
        body.pack(fill="both", expand=True, pady=(16, 0))

        body.columnconfigure(0, weight=1)  # Sidebar
        body.columnconfigure(1, weight=3)  # Carrusel
        body.columnconfigure(2, weight=2)  # Panel info

        # -------- SIDEBAR IZQUIERDA -------- #
        sidebar = tk.Frame(body, bg="#020617", highlightthickness=1, highlightbackground="#1F2937")
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        tk.Label(
            sidebar,
            text="Módulos",
            font=("Segoe UI", 12, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))

        tk.Label(
            sidebar,
            text="Navega por los módulos usando la lista o el carrusel.",
            font=("Segoe UI", 9),
            bg="#020617",
            fg=COLOR_MUTED,
            wraplength=220,
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 10))

        self.sidebar_buttons = []
        for idx, tool in enumerate(TOOLS):
            btn = tk.Button(
                sidebar,
                text=f"{tool['emoji']}  {tool['title']}",
                font=("Segoe UI", 10),
                anchor="w",
                relief="flat",
                bg="#020617",
                fg=COLOR_MUTED,
                activebackground="#111827",
                activeforeground=COLOR_TEXT,
                cursor="hand2",
                command=lambda i=idx: self._select_tool(i),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self.sidebar_buttons.append(btn)

        # -------- CARRUSEL CENTRAL -------- #
        carousel_wrapper = tk.Frame(body, bg=COLOR_BG)
        carousel_wrapper.grid(row=0, column=1, sticky="nsew")

        carousel_wrapper.rowconfigure(0, weight=1)
        carousel_wrapper.columnconfigure(0, weight=1)

        self.carousel_card = tk.Frame(
            carousel_wrapper,
            bg="#020617",
            highlightthickness=1,
            highlightbackground="#1F2937",
        )
        self.carousel_card.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Subcomponentes del carrusel
        self.lbl_emoji = tk.Label(
            self.carousel_card,
            text="",
            font=("Segoe UI Emoji", 34),
            bg="#020617",
            fg=COLOR_TEXT,
        )
        self.lbl_emoji.pack(anchor="nw", padx=20, pady=(18, 4))

        self.lbl_title = tk.Label(
            self.carousel_card,
            text="",
            font=("Segoe UI", 20, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        )
        self.lbl_title.pack(fill="x", padx=20)

        self.lbl_tagline = tk.Label(
            self.carousel_card,
            text="",
            font=("Segoe UI", 10),
            bg="#020617",
            fg=COLOR_MUTED,
            anchor="w",
            wraplength=480,
            justify="left",
        )
        self.lbl_tagline.pack(fill="x", padx=20, pady=(4, 10))

        separator = tk.Frame(self.carousel_card, bg="#111827", height=1)
        separator.pack(fill="x", padx=20, pady=(6, 10))

        self.lbl_desc = tk.Label(
            self.carousel_card,
            text="",
            font=("Segoe UI", 10),
            bg="#020617",
            fg=COLOR_TEXT,
            justify="left",
            wraplength=520,
        )
        self.lbl_desc.pack(fill="x", padx=20)

        # Lista de pasos
        self.steps_frame = tk.Frame(self.carousel_card, bg="#020617")
        self.steps_frame.pack(fill="both", expand=True, padx=20, pady=(8, 0))

        # Footer del carrusel: indicadores + botones
        footer = tk.Frame(self.carousel_card, bg="#020617")
        footer.pack(fill="x", padx=20, pady=(10, 16))

        self.indicadores_frame = tk.Frame(footer, bg="#020617")
        self.indicadores_frame.pack(side="left")

        nav_buttons_frame = tk.Frame(footer, bg="#020617")
        nav_buttons_frame.pack(side="right")

        self.btn_prev = tk.Button(
            nav_buttons_frame,
            text="◀ Anterior",
            font=("Segoe UI", 9),
            bg="#111827",
            fg=COLOR_TEXT,
            relief="flat",
            padx=10,
            cursor="hand2",
            command=self._prev_tool,
        )
        self.btn_prev.pack(side="left", padx=(0, 6))

        self.btn_open = tk.Button(
            nav_buttons_frame,
            text="Abrir módulo",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_ACCENT,
            fg="white",
            relief="flat",
            padx=14,
            cursor="hand2",
            command=self._open_current_tool,
        )
        self.btn_open.pack(side="left", padx=(0, 6))

        self.btn_next = tk.Button(
            nav_buttons_frame,
            text="Siguiente ▶",
            font=("Segoe UI", 9),
            bg="#111827",
            fg=COLOR_TEXT,
            relief="flat",
            padx=10,
            cursor="hand2",
            command=self._next_tool,
        )
        self.btn_next.pack(side="left")

        # -------- PANEL DERECHO (INFO) -------- #
        right_panel = tk.Frame(body, bg="#020617", highlightthickness=1, highlightbackground="#1F2937")
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(14, 0))

        tk.Label(
            right_panel,
            text="Guía rápida",
            font=("Segoe UI", 12, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(12, 4))

        tips = [
            "1. Empieza creando el dataset de estudiantes.",
            "2. Entrena el modelo con suficientes ejemplos.",
            "3. Prueba en vivo la detección de rostros.",
            "4. Usa el diagnóstico para mejorar el modelo.",
        ]
        for t in tips:
            tk.Label(
                right_panel,
                text=t,
                font=("Segoe UI", 9),
                bg="#020617",
                fg=COLOR_MUTED,
                justify="left",
                wraplength=260,
            ).pack(fill="x", padx=16, pady=2)

        separator2 = tk.Frame(right_panel, bg="#111827", height=1)
        separator2.pack(fill="x", padx=16, pady=(10, 10))

        tk.Label(
            right_panel,
            text="Directorio de trabajo:",
            font=("Segoe UI", 9, "bold"),
            bg="#020617",
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x", padx=16)

        tk.Label(
            right_panel,
            text=BASE_DIR,
            font=("Consolas", 8),
            bg="#020617",
            fg=COLOR_MUTED,
            wraplength=260,
            justify="left",
        ).pack(fill="x", padx=16, pady=(0, 10))

    # ---------------- LÓGICA DEL CARRUSEL ---------------- #

    def _render_tool(self):
        tool = TOOLS[self.current_index]

        # Emoji + textos
        self.lbl_emoji.config(text=tool["emoji"])
        self.lbl_title.config(text=tool["title"])
        self.lbl_tagline.config(text=tool["tagline"])
        self.lbl_desc.config(text=tool["desc"])

        # Sidebar: resaltar seleccionado
        for idx, btn in enumerate(self.sidebar_buttons):
            if idx == self.current_index:
                btn.config(bg="#111827", fg=COLOR_TEXT)
            else:
                btn.config(bg="#020617", fg=COLOR_MUTED)

        # Steps
        for w in self.steps_frame.winfo_children():
            w.destroy()

        tk.Label(
            self.steps_frame,
            text="Pasos sugeridos:",
            font=("Segoe UI", 10, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(4, 4))

        for i, step in enumerate(tool["steps"], start=1):
            tk.Label(
                self.steps_frame,
                text=f"{i}. {step}",
                font=("Segoe UI", 9),
                bg="#020617",
                fg=COLOR_MUTED,
                anchor="w",
                justify="left",
                wraplength=520,
            ).pack(fill="x", pady=1)

        # Indicadores inferiores (puntos del carrusel)
        for w in self.indicadores_frame.winfo_children():
            w.destroy()

        for idx, _ in enumerate(TOOLS):
            color = COLOR_ACCENT if idx == self.current_index else "#4B5563"
            dot = tk.Canvas(self.indicadores_frame, width=10, height=10, bg="#020617", highlightthickness=0)
            dot.pack(side="left", padx=3)
            dot.create_oval(2, 2, 8, 8, fill=color, outline=color)

    def _select_tool(self, index: int):
        self.current_index = index
        self._render_tool()

    def _prev_tool(self):
        self.current_index = (self.current_index - 1) % len(TOOLS)
        self._render_tool()

    def _next_tool(self):
        self.current_index = (self.current_index + 1) % len(TOOLS)
        self._render_tool()

    def _open_current_tool(self):
        key = TOOLS[self.current_index]["key"]
        self._run_script(key)

    # ---------------- EJECUCIÓN DE SCRIPTS ---------------- #

    def _run_script(self, key: str) -> None:
        script_name = SCRIPTS.get(key)
        if not script_name:
            messagebox.showerror("Error", "Acción no configurada.")
            return

        script_path = os.path.join(BASE_DIR, script_name)
        if not os.path.exists(script_path):
            messagebox.showerror(
                "Script no encontrado",
                f"No se encontró el script:\n{script_path}\n\n"
                "Verifica que el archivo exista en la carpeta del proyecto.",
            )
            return

        try:
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("Error al ejecutar", str(e))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = MainApp()
    app.run()
