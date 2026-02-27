"""
Archivo: deteccion_tiempo_real.py
Función: Sistema de reconocimiento facial en tiempo real usando cámara web

Nuevo diseño:
- Vista en vivo de lado a lado (ancho completo).
- Controles de cámara (play / stop) integrados en el header de la vista.
- Botón tipo hamburguesa "☰ Resultados" que muestra/oculta un panel flotante con la info en tiempo real.
- Debajo del video, una consola de eventos estilo terminal.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import time
from collections import deque

# PALETA
COLOR_BG = "#020617"
COLOR_CARD = "#020617"
COLOR_BORDER = "#1F2937"
COLOR_PRIMARY = "#1D4ED8"
COLOR_TEXT = "#F9FAFB"
COLOR_MUTED = "#9CA3AF"
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"
COLOR_BTN_START = "#16A34A"
COLOR_BTN_STOP = "#DC2626"
COLOR_TERMINAL_BG = "#020617"

MODELO_PATH = "modelo_estudiantes.h5"
IMG_SIZE = 128


class DeteccionTiempoReal:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FaceLab · Detección en Vivo")
        self.root.geometry("1240x720")
        self.root.configure(bg=COLOR_BG)

        # Variables de lógica
        self.modelo = None
        self.clases = []
        self.cap = None
        self.detectando = False
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.historial_predicciones = {}
        self.max_historial = 7

        self.detecciones_totales = 0
        self.ultima_persona = "Sin detección"
        self.confianza_actual = 0

        # Estado panel resultados
        self.resultados_visible = False

        self.crear_interfaz()
        self.cargar_modelo()

        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_ventana)
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def crear_interfaz(self):
        # HEADER SUPERIOR
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Reconocimiento facial en vivo",
            font=("Segoe UI", 22, "bold"),
            bg=COLOR_PRIMARY,
            fg="white",
        ).pack(side="left", padx=20, pady=14)

        tk.Label(
            header,
            text="Monitorea la cámara y las detecciones en tiempo real.",
            font=("Segoe UI", 10),
            bg=COLOR_PRIMARY,
            fg="#E5E7EB",
        ).pack(side="left", padx=12)

        # CONTENEDOR PRINCIPAL (una sola columna)
        main = tk.Frame(self.root, bg=COLOR_BG)
        main.pack(fill="both", expand=True, padx=20, pady=18)

        main.rowconfigure(0, weight=4)   # video
        main.rowconfigure(1, weight=3)   # terminal
        main.columnconfigure(0, weight=1)

        # ================== VISTA EN VIVO ================== #
        video_card = tk.Frame(main, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        video_card.grid(row=0, column=0, sticky="nsew")

        # --- Barra superior de la vista (título + controles + resultados) --- #
        top_bar = tk.Frame(video_card, bg=COLOR_CARD)
        top_bar.pack(fill="x", pady=(8, 2))

        # Título
        tk.Label(
            top_bar,
            text="Vista en vivo",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
        ).pack(side="left", padx=(14, 6))

        # Pequeña barra de estado del modelo y cámara
        status_bar = tk.Frame(top_bar, bg=COLOR_CARD)
        status_bar.pack(side="left", padx=6)

        self.label_modelo = tk.Label(
            status_bar,
            text="● Modelo no cargado",
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg="#F97316",
        )
        self.label_modelo.pack(side="left", padx=(0, 10))

        self.label_camara = tk.Label(
            status_bar,
            text="● Cámara inactiva",
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
        )
        self.label_camara.pack(side="left")

        # Controles a la derecha: resultados + play/stop
        controls_bar = tk.Frame(top_bar, bg=COLOR_CARD)
        controls_bar.pack(side="right", padx=10)

        # Botón hamburguesa para mostrar resultados en tiempo real
        self.btn_toggle_resultados = tk.Button(
            controls_bar,
            text="☰ Resultados",
            font=("Segoe UI", 9, "bold"),
            bg="#0F172A",
            fg=COLOR_TEXT,
            relief="flat",
            cursor="hand2",
            command=self.toggle_resultados,
        )
        self.btn_toggle_resultados.pack(side="right", padx=(4, 0))

        # Botones de cámara: solo íconos
        self.btn_parar = tk.Button(
            controls_bar,
            text="■",
            font=("Segoe UI", 10, "bold"),
            width=3,
            bg=COLOR_BTN_STOP,
            fg="white",
            relief="flat",
            cursor="hand2",
            state="disabled",
            command=self.parar_deteccion,
        )
        self.btn_parar.pack(side="right", padx=(4, 0))

        self.btn_iniciar = tk.Button(
            controls_bar,
            text="▶",
            font=("Segoe UI", 10, "bold"),
            width=3,
            bg=COLOR_BTN_START,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self.iniciar_deteccion,
        )
        self.btn_iniciar.pack(side="right", padx=(4, 0))

        # --- Canvas de video (ancho completo) --- #
        self.canvas_video = tk.Canvas(
            video_card,
            bg="black",
            highlightthickness=0,
            width=1100,
            height=420,
        )
        self.canvas_video.pack(expand=True, fill="both", padx=16, pady=(4, 12))

        # ================== CONSOLA / TERMINAL ================== #
        terminal_card = tk.Frame(main, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        terminal_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        header_terminal = tk.Frame(terminal_card, bg=COLOR_CARD)
        header_terminal.pack(fill="x", pady=(8, 0))

        tk.Label(
            header_terminal,
            text="Consola de eventos",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
        ).pack(side="left", padx=14)

        tk.Label(
            header_terminal,
            text="(mensajes del sistema de detección)",
            font=("Segoe UI", 8),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
        ).pack(side="left")

        # Área tipo terminal
        self.text_log = scrolledtext.ScrolledText(
            terminal_card,
            font=("Consolas", 9),
            bg=COLOR_TERMINAL_BG,
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            wrap=tk.WORD,
        )
        self.text_log.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        # ================== PANEL FLOTANTE DE RESULTADOS ================== #
        # Se muestra/oculta con el botón "☰ Resultados"
        self.panel_resultados = tk.Frame(self.root, bg="#020617", highlightthickness=1, highlightbackground="#4B5563")

        tk.Label(
            self.panel_resultados,
            text="Resultados en tiempo real",
            font=("Segoe UI", 11, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 4))

        # Línea estado general
        tk.Label(
            self.panel_resultados,
            text="Resumen de la última detección",
            font=("Segoe UI", 9),
            bg="#020617",
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))

        # Última detección
        block_det = tk.Frame(self.panel_resultados, bg="#020617")
        block_det.pack(fill="x", padx=14, pady=(2, 8))

        tk.Label(
            block_det,
            text="Persona detectada:",
            font=("Segoe UI", 9),
            bg="#020617",
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x")

        self.label_persona = tk.Label(
            block_det,
            text="—",
            font=("Segoe UI", 11, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        )
        self.label_persona.pack(fill="x", pady=(0, 4))

        tk.Label(
            block_det,
            text="Confianza:",
            font=("Segoe UI", 9),
            bg="#020617",
            fg=COLOR_MUTED,
            anchor="w",
        ).pack(fill="x")

        self.label_confianza = tk.Label(
            block_det,
            text="0 %",
            font=("Segoe UI", 10),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        )
        self.label_confianza.pack(fill="x")

        self.progressbar_confianza = ttk.Progressbar(
            block_det,
            orient="horizontal",
            mode="determinate",
            length=220,
        )
        self.progressbar_confianza.pack(fill="x", pady=(4, 4))

        # Estadísticas
        block_stats = tk.Frame(self.panel_resultados, bg="#020617")
        block_stats.pack(fill="x", padx=14, pady=(4, 10))

        tk.Label(
            block_stats,
            text="Estadísticas de sesión",
            font=("Segoe UI", 9, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 2))

        self.label_detecciones = tk.Label(
            block_stats,
            text="Detecciones: 0",
            font=("Segoe UI", 9),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        )
        self.label_detecciones.pack(fill="x")

        self.label_fps = tk.Label(
            block_stats,
            text="FPS: 0",
            font=("Segoe UI", 9),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        )
        self.label_fps.pack(fill="x", pady=(0, 4))

        tk.Label(
            self.panel_resultados,
            text="Tip: abre este panel solo cuando quieras revisar los valores, para dejar más espacio al video.",
            font=("Segoe UI", 8),
            bg="#020617",
            fg=COLOR_MUTED,
            wraplength=260,
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 10))

        # Log inicial
        self.agregar_log("✅ Sistema de detección iniciado")

    # Mostrar / ocultar panel flotante de resultados
    def toggle_resultados(self):
        if self.resultados_visible:
            self.panel_resultados.place_forget()
            self.resultados_visible = False
        else:
            # Lo colocamos como una "pestaña" flotante en la esquina superior derecha
            self.panel_resultados.place(relx=1.0, rely=0.22, anchor="ne", width=290, height=260)
            self.resultados_visible = True

    # ------------------------------------------------------------------
    # LÓGICA (igual que antes, solo adaptada a la nueva UI)
    # ------------------------------------------------------------------
    def agregar_log(self, mensaje):
        timestamp = time.strftime("%H:%M:%S")
        self.text_log.insert(tk.END, f"[{timestamp}] {mensaje}\n")
        self.text_log.see(tk.END)

    def cargar_modelo(self):
        if not os.path.exists(MODELO_PATH):
            self.agregar_log("❌ Modelo no encontrado")
            self.label_modelo.config(text="● Modelo no encontrado", fg=COLOR_DANGER)
            messagebox.showwarning(
                "Advertencia",
                "No se encontró modelo entrenado.\nEntrena un modelo primero."
            )
            return False

        try:
            self.modelo = load_model(MODELO_PATH)
            data_dir = "data_set_estudiantes"
            if os.path.exists(data_dir):
                self.clases = sorted(
                    [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                )

            self.agregar_log(f"✅ Modelo cargado: {len(self.clases)} clases")
            if self.clases:
                self.agregar_log(f"  Clases: {', '.join(self.clases)}")
            self.label_modelo.config(text="● Modelo cargado", fg=COLOR_SUCCESS)
            return True

        except Exception as e:
            self.agregar_log(f"❌ Error al cargar modelo: {str(e)}")
            messagebox.showerror("Error", f"Error al cargar modelo:\n{str(e)}")
            return False

    def suavizado_temporal(self, rostro_id, clase_idx, confianza):
        if rostro_id not in self.historial_predicciones:
            self.historial_predicciones[rostro_id] = deque(maxlen=self.max_historial)

        self.historial_predicciones[rostro_id].append((clase_idx, confianza))

        if len(self.historial_predicciones[rostro_id]) >= 5:
            votos = {}
            for idx, conf in self.historial_predicciones[rostro_id]:
                if idx not in votos:
                    votos[idx] = []
                votos[idx].append(conf)

            clase_ganadora = max(votos, key=lambda k: len(votos[k]))
            confianza_promedio = sum(votos[clase_ganadora]) / len(votos[clase_ganadora])
            confianza_calibrada = (confianza_promedio ** 0.85) * 100

            return clase_ganadora, confianza_calibrada
        else:
            confianza_calibrada = (confianza ** 0.85) * 100
            return clase_idx, confianza_calibrada

    def iniciar_deteccion(self):
        if self.modelo is None:
            messagebox.showwarning("Advertencia", "Carga el modelo primero")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara")
            return

        self.detectando = True
        self.btn_iniciar.config(state='disabled')
        self.btn_parar.config(state='normal')
        self.label_camara.config(text="● Cámara activa", fg=COLOR_SUCCESS)
        self.agregar_log("🎥 Detección iniciada")

        self.historial_predicciones = {}
        threading.Thread(target=self.proceso_deteccion, daemon=True).start()

    def proceso_deteccion(self):
        frame_count = 0
        start_time = time.time()

        while self.detectando:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                self.historial_predicciones = {}

            for i, (x, y, w, h) in enumerate(faces):
                rostro = frame[y:y + h, x:x + w]
                rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
                rostro_resized = cv2.resize(rostro_rgb, (IMG_SIZE, IMG_SIZE))
                rostro_normalized = rostro_resized / 255.0
                rostro_input = np.expand_dims(rostro_normalized, axis=0)

                prediccion = self.modelo.predict(rostro_input, verbose=0)

                confianzas_ordenadas = np.sort(prediccion[0])[::-1]
                primera = confianzas_ordenadas[0]
                segunda = confianzas_ordenadas[1] if len(confianzas_ordenadas) > 1 else 0
                diferencia = primera - segunda

                clase_idx = np.argmax(prediccion)
                confianza_raw = prediccion[0][clase_idx]

                if diferencia < 0.10:
                    confianza_raw = 0.35

                rostro_id = f"rostro_{i}"
                clase_idx_suavizada, confianza = self.suavizado_temporal(
                    rostro_id, clase_idx, confianza_raw
                )

                if confianza < 50 and rostro_id in self.historial_predicciones:
                    self.historial_predicciones[rostro_id].clear()

                if confianza > 62:
                    nombre = self.clases[clase_idx_suavizada] if self.clases else "Desconocido"
                    color = (0, 255, 0)
                    self.detecciones_totales += 1
                    self.ultima_persona = nombre
                    self.confianza_actual = confianza
                else:
                    nombre = "Desconocido"
                    color = (0, 0, 255)
                    self.confianza_actual = confianza

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)

                cv2.putText(
                    frame,
                    nombre,
                    (x + 5, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{confianza:.1f}%",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                self.root.after(0, self.actualizar_interfaz, nombre, confianza)

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                self.root.after(0, lambda: self.label_fps.config(text=f"FPS: {fps:.1f}"))
                frame_count = 0
                start_time = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((1100, 420))
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, self._actualizar_canvas, imgtk)

            time.sleep(0.05)

        if self.cap:
            self.cap.release()

    def _actualizar_canvas(self, imgtk):
        self.canvas_video.delete("all")
        self.canvas_video.create_image(550, 210, image=imgtk)
        self.canvas_video.imgtk = imgtk

    def actualizar_interfaz(self, nombre, confianza):
        self.label_persona.config(text=nombre)
        self.label_confianza.config(text=f"{confianza:.1f}%")
        self.progressbar_confianza['value'] = confianza

        if confianza >= 70:
            self.label_confianza.config(fg=COLOR_SUCCESS)
        elif confianza >= 62:
            self.label_confianza.config(fg=COLOR_WARNING)
        else:
            self.label_confianza.config(fg=COLOR_DANGER)

        self.label_detecciones.config(text=f"Detecciones: {self.detecciones_totales}")

    def parar_deteccion(self):
        self.detectando = False
        self.btn_iniciar.config(state='normal')
        self.btn_parar.config(state='disabled')
        self.label_camara.config(text="● Cámara inactiva", fg=COLOR_MUTED)
        self.agregar_log("⏹️ Detección detenida")
        self.agregar_log(f"  Total detecciones: {self.detecciones_totales}")

    def cerrar_ventana(self):
        self.detectando = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    DeteccionTiempoReal()
