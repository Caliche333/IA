"""
Archivo: captura_dataset.py
Función: Interfaz para captura y creación del dataset de estudiantes con cámara web y bounding boxes.
Nuevo diseño: layout en dos columnas + sección de tips tipo “acordeón”.
"""

import cv2
import os
from tkinter import *
from tkinter import ttk, messagebox

COLOR_DARK = "#020617"
COLOR_TEXT = "#F9FAFB"
COLOR_SUCCESS = "#6366F1"
COLOR_MUTED = "#9CA3AF"
COLOR_CARD = "#020617"
COLOR_BORDER = "#1F2937"

DATA_DIR = "data_set_estudiantes"
NUM_FOTOS = 100


class CapturaDataset:
    def __init__(self):
        self.root = Tk()
        self.root.title("FaceLab · Captura de Dataset")
        self.root.geometry("1100x620")
        self.root.configure(bg=COLOR_DARK)

        self.nombres = []
        self.tips_expandidos = False

        # CONTENEDOR PRINCIPAL
        main = Frame(self.root, bg=COLOR_DARK)
        main.pack(fill="both", expand=True, padx=24, pady=24)

        # HEADER SUPERIOR
        header = Frame(main, bg="#0B1120", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        Label(
            header,
            text="Captura de dataset de estudiantes",
            font=("Segoe UI", 22, "bold"),
            bg="#0B1120",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(side="left", padx=20, pady=10)

        Label(
            header,
            text="Registra estudiantes, genera sus carpetas y captura automáticamente sus rostros.",
            font=("Segoe UI", 10),
            bg="#0B1120",
            fg="#E5E7EB",
            anchor="w",
        ).pack(side="left", padx=10)

        # CUERPO EN DOS COLUMNAS
        body = Frame(main, bg=COLOR_DARK)
        body.pack(fill="both", expand=True, pady=(16, 0))

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)

        # -------- COLUMNA IZQUIERDA: FORMULARIO -------- #
        left = Frame(body, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        Label(
            left,
            text="Nuevo estudiante",
            font=("Segoe UI", 13, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 4))

        Label(
            left,
            text="Escribe el nombre exactamente como quieres que se cree la carpeta del estudiante.",
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            wraplength=320,
            justify="left",
        ).pack(fill="x", padx=18, pady=(0, 10))

        self.nombre_entry = Entry(left, font=("Segoe UI", 12))
        self.nombre_entry.pack(fill="x", padx=18, pady=(0, 8))

        self.btn_agregar = Button(
            left,
            text="➕ Agregar a la lista",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            activebackground=COLOR_SUCCESS,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.agregar_nombre,
        )
        self.btn_agregar.pack(fill="x", padx=18, pady=(4, 10))

        # Tarjeta de resumen rápido
        resumen_card = Frame(left, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        resumen_card.pack(fill="x", padx=18, pady=(6, 10))

        Label(
            resumen_card,
            text="Resumen de captura",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))

        self.lbl_total_estudiantes = Label(
            resumen_card,
            text="Estudiantes en la lista: 0",
            font=("Segoe UI", 10),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            anchor="w",
        )
        self.lbl_total_estudiantes.pack(fill="x", padx=12, pady=(2, 2))

        Label(
            resumen_card,
            text=f"Fotos por estudiante: {NUM_FOTOS}",
            font=("Segoe UI", 10),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 6))

        Label(
            resumen_card,
            text="Las imágenes se guardarán en:",
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(4, 0))

        Label(
            resumen_card,
            text=DATA_DIR,
            font=("Consolas", 8),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            anchor="w",
            wraplength=320,
            justify="left",
        ).pack(fill="x", padx=12, pady=(0, 10))

        # Botón principal de captura (full width al final)
        self.btn_iniciar = Button(
            left,
            text="📸 Iniciar captura de imágenes",
            font=("Segoe UI", 13, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            activebackground=COLOR_SUCCESS,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.iniciar_captura,
        )
        self.btn_iniciar.pack(fill="x", padx=18, pady=(4, 18))

        # -------- COLUMNA DERECHA: LISTA + TIPS -------- #
        right = Frame(body, bg=COLOR_DARK)
        right.grid(row=0, column=1, sticky="nsew")

        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)

        # Tarjeta lista de estudiantes
        card_list = Frame(right, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        card_list.grid(row=0, column=0, sticky="nsew")

        Label(
            card_list,
            text="Estudiantes preparados para captura",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(12, 4))

        Label(
            card_list,
            text="Puedes agregar varios estudiantes antes de iniciar la sesión de captura.",
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 6))

        list_container = Frame(card_list, bg=COLOR_CARD)
        list_container.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        scrollbar = Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.listbox_nombres = Listbox(
            list_container,
            font=("Segoe UI", 11),
            height=8,
            bg="#020617",
            fg=COLOR_TEXT,
            selectbackground=COLOR_SUCCESS,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        self.listbox_nombres.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox_nombres.yview)

        # Tarjeta de tips tipo acordeón
        tips_wrapper = Frame(right, bg=COLOR_DARK)
        tips_wrapper.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        tips_header = Frame(tips_wrapper, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        tips_header.pack(fill="x")

        self.btn_toggle_tips = Button(
            tips_header,
            text="▶ Consejos de captura",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            relief="flat",
            anchor="w",
            cursor="hand2",
            command=self.toggle_tips,
        )
        self.btn_toggle_tips.pack(fill="x", padx=12, pady=8)

        self.tips_body = Frame(
            tips_wrapper,
            bg=COLOR_CARD,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        # Inicialmente oculto, se muestra al pulsar el botón

        self.root.mainloop()

    # -------- LÓGICA UI EXTRA (ACORDEÓN) -------- #

    def toggle_tips(self):
        if self.tips_expandidos:
            # Ocultar
            self.tips_body.forget()
            self.tips_expandidos = False
            self.btn_toggle_tips.config(text="▶ Consejos de captura")
        else:
            # Mostrar
            self.tips_body.pack(fill="both", expand=True, pady=(0, 0))
            self.tips_expandidos = True
            self.btn_toggle_tips.config(text="▼ Consejos de captura")
            self._render_tips()

    def _render_tips(self):
        # Limpiar contenido
        for w in self.tips_body.winfo_children():
            w.destroy()

        Label(
            self.tips_body,
            text=(
                "• Procura buena iluminación, sin sombras fuertes en el rostro.\n"
                "• El estudiante debe mirar de frente y también girar ligeramente la cabeza.\n"
                "• Evita gafas oscuras o elementos que tapen gran parte de la cara.\n"
                f"• Se capturan automáticamente {NUM_FOTOS} imágenes por estudiante.\n"
                "• Puedes repetir la captura en otro momento para actualizar el dataset."
            ),
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            justify="left",
            wraplength=540,
        ).pack(fill="both", expand=True, padx=14, pady=10)

    # -------- LÓGICA ORIGINAL (SIN CAMBIOS FUNCIONALES) -------- #

    def agregar_nombre(self):
        nombre = self.nombre_entry.get().strip()
        if nombre and nombre not in self.nombres:
            self.nombres.append(nombre)
            self.listbox_nombres.insert(END, nombre)
            self.nombre_entry.delete(0, END)
            self.lbl_total_estudiantes.config(
                text=f"Estudiantes en la lista: {len(self.nombres)}"
            )

    def iniciar_captura(self):
        if len(self.nombres) < 1:
            messagebox.showwarning("Warning", "Agregue al menos un nombre para capturar fotos")
            return
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        for nombre in self.nombres:
            carpeta = os.path.join(DATA_DIR, nombre)
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)

            contador = 0
            while contador < NUM_FOTOS:
                ret, frame = cam.read()
                if not ret:
                    continue
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                caras = face_cascade.detectMultiScale(gris, 1.1, 5)
                for (x, y, w, h) in caras:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    if contador < NUM_FOTOS:
                        cv2.imwrite(
                            os.path.join(carpeta, f"img_{contador:03d}.jpg"),
                            frame[y:y + h, x:x + w],
                        )
                        contador += 1

                        cv2.imwrite(
                            os.path.join(carpeta, f"img_{contador:03d}.jpg"),
                            frame[y:y + h, x:x + w],
                        )
                        contador += 1
                        cv2.waitKey(300)

                cv2.putText(
                    frame,
                    f"Capturando {nombre}: {contador}/{NUM_FOTOS}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Captura Dataset", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Info", "Captura finalizada")


if __name__ == '__main__':
    CapturaDataset()
