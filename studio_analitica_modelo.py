"""
Archivo: diagnostico_modelo.py
Función: Diagnostica el modelo entrenado mostrando métricas, matriz de confusión y gráficos de rendimiento
Nuevo diseño: layout de 3 columnas (sidebar acciones, centro de gráficos, log a la derecha).
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import threading

COLOR_DARK = "#020617"
COLOR_TEXT = "#F9FAFB"
COLOR_SUCCESS = "#6366F1"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"
DATA_DIR = "data_set_estudiantes"
MODELO_PATH = "modelo_estudiantes.h5"


class VentanaDiagnostico:
    def __init__(self, parent, titulo):
        self.window = tk.Toplevel(parent)
        self.window.title(titulo)
        self.window.geometry("720x260")
        self.window.configure(bg=COLOR_DARK)
        self.window.transient(parent)
        self.window.grab_set()

        # Centrar ventana
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - 300
        y = (self.window.winfo_screenheight() // 2) - 100
        self.window.geometry(f'+{x}+{y}')

        tk.Label(
            self.window,
            text=titulo,
            font=("Segoe UI", 16, "bold"),
            bg=COLOR_DARK,
            fg=COLOR_TEXT
        ).pack(pady=15)

        self.label_estado = tk.Label(
            self.window,
            text="Analizando modelo...",
            font=("Segoe UI", 12),
            bg=COLOR_DARK,
            fg=COLOR_TEXT
        )
        self.label_estado.pack(pady=10)

        self.progressbar = ttk.Progressbar(self.window, length=500, mode='indeterminate')
        self.progressbar.pack(pady=15)
        self.progressbar.start(10)

    def actualizar(self, mensaje):
        self.label_estado.config(text=mensaje)
        self.window.update()

    def cerrar(self):
        self.progressbar.stop()
        self.window.destroy()


class DiagnosticoModelo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FaceLab · Diagnóstico del Modelo")
        self.root.geometry("1220x720")
        self.root.configure(bg=COLOR_DARK)

        # Variables
        self.modelo = None
        self.dataset = None
        self.clases = []
        self.y_true = []
        self.y_pred = []
        self.accuracy = 0

        # CONTENEDOR PRINCIPAL
        main = tk.Frame(self.root, bg=COLOR_DARK)
        main.pack(fill="both", expand=True, padx=24, pady=24)

        # ENCABEZADO SUPERIOR
        header = tk.Frame(main, bg="#0B1120", height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Panel de diagnóstico del modelo",
            font=("Segoe UI", 22, "bold"),
            bg="#0B1120",
            fg="white",
        ).pack(side="left", padx=20, pady=15)

        tk.Label(
            header,
            text="Evalúa accuracy, matriz de confusión y reporte completo por clase.",
            font=("Segoe UI", 10),
            bg="#0B1120",
            fg="#E5E7EB",
        ).pack(side="left", padx=12)

        # CUERPO PRINCIPAL EN 3 COLUMNAS
        body = tk.Frame(main, bg=COLOR_DARK)
        body.pack(fill="both", expand=True, pady=(16, 0))

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=1)

        # -------- COLUMNA IZQUIERDA: ESTADO + ACCIONES -------- #
        left = tk.Frame(body, bg="#020617", highlightthickness=1, highlightbackground="#1F2937")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        tk.Label(
            left,
            text="Estado del modelo",
            font=("Segoe UI", 12, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 6))

        self.label_modelo_estado = tk.Label(
            left,
            text="⚪ Modelo no cargado",
            font=("Segoe UI", 10),
            bg="#020617",
            fg="#F97316",
            anchor="w",
        )
        self.label_modelo_estado.pack(fill="x", padx=16)

        self.label_estado = tk.Label(
            left,
            text="Esperando análisis...",
            font=("Segoe UI", 10),
            bg="#020617",
            fg="#9CA3AF",
            anchor="w",
        )
        self.label_estado.pack(fill="x", padx=16, pady=(4, 10))

        progress_frame = tk.Frame(left, bg="#020617")
        progress_frame.pack(fill="x", padx=16, pady=(0, 10))

        self.progressbar = ttk.Progressbar(progress_frame, mode="determinate")
        self.progressbar.pack(fill="x")

        # Botones de análisis en una "barra vertical"
        actions = tk.Frame(left, bg="#020617")
        actions.pack(fill="x", padx=14, pady=(6, 10))

        tk.Label(
            actions,
            text="Acciones rápidas",
            font=("Segoe UI", 10, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        tk.Button(
            actions,
            text="📊 Analizar modelo completo",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self.analizar_modelo,
        ).pack(fill="x", pady=(0, 6))

        tk.Button(
            actions,
            text="📈 Ver métricas generales",
            font=("Segoe UI", 10),
            bg="#111827",
            fg=COLOR_TEXT,
            relief="flat",
            cursor="hand2",
            command=self.mostrar_metricas,
        ).pack(fill="x", pady=(0, 4))

        tk.Button(
            actions,
            text="🧩 Matriz de confusión",
            font=("Segoe UI", 10),
            bg="#111827",
            fg=COLOR_TEXT,
            relief="flat",
            cursor="hand2",
            command=self.mostrar_confusion_matrix,
        ).pack(fill="x", pady=(0, 4))

        tk.Button(
            actions,
            text="📄 Reporte detallado",
            font=("Segoe UI", 10),
            bg="#111827",
            fg=COLOR_TEXT,
            relief="flat",
            cursor="hand2",
            command=self.mostrar_reporte,
        ).pack(fill="x", pady=(0, 4))

        # -------- COLUMNA CENTRAL: GRÁFICOS / RESULTADOS -------- #
        self.frame_right = tk.Frame(body, bg=COLOR_DARK)
        self.frame_right.grid(row=0, column=1, sticky="nsew")

        placeholder = tk.Label(
            self.frame_right,
            text="Aquí se mostrarán las métricas gráficas\n(matriz de confusión, accuracy, etc.).",
            font=("Segoe UI", 11),
            bg=COLOR_DARK,
            fg="#9CA3AF",
            justify="center",
        )
        placeholder.pack(expand=True)

        self.canvas_graficos = None

        # -------- COLUMNA DERECHA: LOG / INFORMACIÓN -------- #
        right = tk.Frame(body, bg="#020617", highlightthickness=1, highlightbackground="#1F2937")
        right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))

        tk.Label(
            right,
            text="Registro de eventos",
            font=("Segoe UI", 12, "bold"),
            bg="#020617",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 6))

        self.text_info = scrolledtext.ScrolledText(
            right,
            font=("Consolas", 8),
            bg="#020617",
            fg=COLOR_TEXT,
            wrap=tk.WORD,
        )
        self.text_info.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Mensaje inicial y verificación de modelo
        self.agregar_log("Sistema de diagnóstico iniciado")
        self.verificar_modelo()

        self.root.mainloop()

    # ---------------- LÓGICA ---------------- #

    def verificar_modelo(self):
        """Verifica si existe el modelo."""
        if os.path.exists(MODELO_PATH):
            self.agregar_log(f"Modelo encontrado: {MODELO_PATH}")
            self.label_modelo_estado.config(text="✅ Modelo disponible", fg=COLOR_SUCCESS)
        else:
            self.agregar_log(f"No se encontró el modelo: {MODELO_PATH}")
            self.label_modelo_estado.config(text="Modelo no encontrado", fg=COLOR_DANGER)
            messagebox.showwarning(
                "Advertencia",
                "No se encontró modelo entrenado.\nEntrena un modelo primero."
            )

    def agregar_log(self, mensaje):
        """Añade mensaje al log."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        self.text_info.insert(tk.END, f"[{timestamp}] {mensaje}\n")
        self.text_info.see(tk.END)

    def analizar_modelo(self):
        """Analiza el modelo completo."""
        if not os.path.exists(MODELO_PATH):
            messagebox.showerror("Error", "No existe modelo entrenado")
            return

        if not os.path.exists(DATA_DIR):
            messagebox.showerror("Error", "No existe dataset de prueba")
            return

        ventana = VentanaDiagnostico(self.root, "Analizando Modelo")

        def proceso_analisis():
            try:
                # Cargar modelo
                ventana.actualizar("Cargando modelo...")
                self.modelo = load_model(MODELO_PATH)
                self.agregar_log("Modelo cargado correctamente")

                # Cargar dataset
                ventana.actualizar("Cargando dataset de prueba...")
                self.dataset = tf.keras.utils.image_dataset_from_directory(
                    DATA_DIR,
                    image_size=(128, 128),
                    batch_size=32,
                    shuffle=False
                )

                self.clases = self.dataset.class_names
                self.agregar_log(f"Dataset cargado: {len(self.clases)} clases")
                self.agregar_log(f"   Clases: {', '.join(self.clases)}")

                # Realizar predicciones
                ventana.actualizar("Realizando predicciones...")
                self.y_true = []
                self.y_pred = []

                for images, labels in self.dataset:
                    predictions = self.modelo.predict(images, verbose=0)
                    self.y_pred.extend(np.argmax(predictions, axis=1))
                    self.y_true.extend(labels.numpy())

                # Calcular accuracy
                self.accuracy = accuracy_score(self.y_true, self.y_pred) * 100

                ventana.cerrar()

                self.agregar_log(" Análisis completado")
                self.agregar_log(f"   Accuracy: {self.accuracy:.2f}%")
                self.agregar_log(f"   Total muestras: {len(self.y_true)}")

                messagebox.showinfo(
                    "Éxito",
                    f"Análisis completado\n\nAccuracy: {self.accuracy:.2f}%"
                )

            except Exception as e:
                ventana.cerrar()
                self.agregar_log(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", f"Error en análisis:\n{str(e)}")

        threading.Thread(target=proceso_analisis, daemon=True).start()

    def mostrar_metricas(self):
        """Muestra métricas del modelo."""
        if self.modelo is None:
            messagebox.showwarning("Advertencia", "Analiza el modelo primero")
            return

        # Limpiar frame central
        for widget in self.frame_right.winfo_children():
            widget.destroy()

        tk.Label(
            self.frame_right,
            text="MÉTRICAS DEL MODELO",
            font=("Segoe UI", 18, "bold"),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
        ).pack(pady=15)

        frame_metricas = tk.Frame(self.frame_right, bg=COLOR_DARK)
        frame_metricas.pack(expand=True, fill='both', padx=20)

        # Accuracy
        frame_acc = tk.Frame(frame_metricas, bg="#1E3A8A", relief='raised', borderwidth=2)
        frame_acc.pack(fill='x', pady=10)

        tk.Label(
            frame_acc,
            text="ACCURACY GENERAL",
            font=("Segoe UI", 14, "bold"),
            bg="#1E3A8A",
            fg="white",
        ).pack(pady=10)

        tk.Label(
            frame_acc,
            text=f"{self.accuracy:.2f}%",
            font=("Segoe UI", 48, "bold"),
            bg="#1E3A8A",
            fg="#6366F1",
        ).pack(pady=20)

        # Información del modelo
        frame_info_modelo = tk.LabelFrame(
            frame_metricas,
            text=" Información del Modelo ",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
        )
        frame_info_modelo.pack(fill='x', pady=10)

        info_text = f"""
Número de clases: {len(self.clases)}
Total de muestras evaluadas: {len(self.y_true)}
Arquitectura: CNN (Convolutional Neural Network)

Clases detectadas:
{chr(10).join([f'  • {clase}' for clase in self.clases])}
        """

        tk.Label(
            frame_info_modelo,
            text=info_text,
            font=("Segoe UI", 11),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            justify='left',
        ).pack(padx=20, pady=15)

        self.agregar_log("📊 Métricas mostradas")

    def mostrar_confusion_matrix(self):
        """Muestra matriz de confusión."""
        if self.modelo is None or len(self.y_true) == 0:
            messagebox.showwarning("Advertencia", "Analiza el modelo primero")
            return

        for widget in self.frame_right.winfo_children():
            widget.destroy()

        tk.Label(
            self.frame_right,
            text="MATRIZ DE CONFUSIÓN",
            font=("Segoe UI", 18, "bold"),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
        ).pack(pady=15)

        fig = Figure(figsize=(6, 4), facecolor=COLOR_DARK)
        ax = fig.add_subplot(111)

        cm = confusion_matrix(self.y_true, self.y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.clases,
            yticklabels=self.clases,
            ax=ax,
            cbar_kws={'label': 'Cantidad'}
        )

        ax.set_xlabel('Predicción', fontsize=12, color=COLOR_TEXT)
        ax.set_ylabel('Real', fontsize=12, color=COLOR_TEXT)
        ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold', color=COLOR_TEXT)

        ax.tick_params(colors=COLOR_TEXT)
        ax.xaxis.label.set_color(COLOR_TEXT)
        ax.yaxis.label.set_color(COLOR_TEXT)
        ax.title.set_color(COLOR_TEXT)

        fig.patch.set_facecolor(COLOR_DARK)
        ax.set_facecolor(COLOR_DARK)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_right)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', padx=20, pady=10)

        self.agregar_log("🎯 Matriz de confusión mostrada")

    def mostrar_reporte(self):
        """Muestra reporte detallado."""
        if self.modelo is None or len(self.y_true) == 0:
            messagebox.showwarning("Advertencia", "Analiza el modelo primero")
            return

        for widget in self.frame_right.winfo_children():
            widget.destroy()

        tk.Label(
            self.frame_right,
            text="REPORTE DETALLADO",
            font=("Segoe UI", 18, "bold"),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
        ).pack(pady=15)

        etiquetas_presentes = sorted(list(set(self.y_true) | set(self.y_pred)))
        try:
            if len(self.clases) == len(etiquetas_presentes):
                reporte = classification_report(
                    self.y_true,
                    self.y_pred,
                    labels=etiquetas_presentes,
                    target_names=[self.clases[i] for i in etiquetas_presentes],
                    digits=4,
                )
            else:
                reporte = classification_report(
                    self.y_true,
                    self.y_pred,
                    labels=etiquetas_presentes,
                    digits=4,
                )
        except Exception as e:
            reporte = f"No se pudo generar el reporte de clasificación:\n{e}"

        text_reporte = scrolledtext.ScrolledText(
            self.frame_right,
            font=("Courier New", 11),
            bg="#020617",
            fg=COLOR_TEXT,
            wrap=tk.WORD,
        )
        text_reporte.pack(fill="both", expand=True, padx=20, pady=10)

        text_reporte.insert(tk.END, "=" * 70 + "\n")
        text_reporte.insert(tk.END, "REPORTE DE CLASIFICACIÓN\n")
        text_reporte.insert(tk.END, "=" * 70 + "\n\n")
        text_reporte.insert(tk.END, reporte)
        text_reporte.insert(tk.END, "\n" + "=" * 70 + "\n")
        text_reporte.insert(tk.END, f"\nACCURACY GENERAL: {self.accuracy:.2f}%\n")
        text_reporte.insert(tk.END, f"TOTAL DE MUESTRAS: {len(self.y_true)}\n")
        text_reporte.insert(tk.END, "=" * 70 + "\n")

        text_reporte.config(state='disabled')

        self.agregar_log("📋 Reporte detallado mostrado")


if __name__ == '__main__':
    DiagnosticoModelo()
