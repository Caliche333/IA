"""
Archivo: entrenar_modelo.py
Función: Entrena modelo CNN con dataset almacenado, con interfaz para ajustar hiperparámetros
Nuevo diseño: panel de entrenamiento con tarjetas de hiperparámetros.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2  # (Por si luego quieres añadir regularización)
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# CONSTANTES DE COLORES
COLOR_DARK = "#020617"
COLOR_TEXT = "#F9FAFB"
COLOR_SUCCESS = "#6366F1"
COLOR_MUTED = "#9CA3AF"
COLOR_CARD = "#020617"
COLOR_BORDER = "#1F2937"

DATA_DIR = "data_set_estudiantes"


class VentanaProgreso:
    def __init__(self, parent, titulo, mensaje, determinado=False):
        self.window = tk.Toplevel(parent)
        self.window.title(titulo)
        self.window.geometry("640x220")
        self.window.configure(bg=COLOR_DARK)
        self.window.transient(parent)
        self.window.grab_set()
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - 275
        y = (self.window.winfo_screenheight() // 2) - 100
        self.window.geometry(f'+{x}+{y}')

        tk.Label(
            self.window,
            text=titulo,
            font=("Segoe UI", 16, "bold"),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
        ).pack(pady=15)

        self.label_mensaje = tk.Label(
            self.window,
            text=mensaje,
            font=("Segoe UI", 11),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
        )
        self.label_mensaje.pack(pady=10)

        self.determinado = determinado
        if determinado:
            self.progressbar = ttk.Progressbar(
                self.window,
                length=450,
                mode='determinate',
                maximum=100,
            )
        else:
            self.progressbar = ttk.Progressbar(
                self.window,
                length=450,
                mode='indeterminate',
            )
        self.progressbar.pack(pady=15)
        if not determinado:
            self.progressbar.start(10)

        self.label_estado = tk.Label(
            self.window,
            text="",
            font=("Segoe UI", 9),
            bg=COLOR_DARK,
            fg=COLOR_MUTED,
        )
        self.label_estado.pack(pady=5)

    def actualizar(self, mensaje=None, estado=None, progreso=None):
        if mensaje:
            self.label_mensaje.config(text=mensaje)
        if estado:
            self.label_estado.config(text=estado)
        if progreso is not None and self.determinado:
            self.progressbar['value'] = progreso
        self.window.update()

    def cerrar(self):
        if not self.determinado:
            self.progressbar.stop()
        self.window.destroy()


class CallbackGUI(Callback):
    def __init__(self, ventana_progreso, total_epochs):
        super().__init__()
        self.ventana = ventana_progreso
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progreso = ((epoch + 1) / self.total_epochs) * 100
        acc = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        self.ventana.actualizar(
            mensaje=f"Entrenando... Epoch {epoch + 1}/{self.total_epochs}",
            estado=f"Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}%",
            progreso=progreso,
        )


class EntrenamientoModelo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FaceLab · Entrenamiento de Modelo")
        self.root.geometry("1040x620")
        self.root.configure(bg=COLOR_DARK)

        # Variables de hiperparámetros
        self.epochs = tk.IntVar(value=50)
        self.batch_size = tk.IntVar(value=6)
        self.lr_scale_var = tk.IntVar(value=5)

        # CONTENEDOR PRINCIPAL
        main = tk.Frame(self.root, bg=COLOR_DARK)
        main.pack(fill="both", expand=True, padx=24, pady=24)

        # ENCABEZADO
        header = tk.Frame(main, bg="#0B1120", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Entrenamiento del modelo CNN",
            font=("Segoe UI", 22, "bold"),
            bg="#0B1120",
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(side="left", padx=20, pady=12)

        tk.Label(
            header,
            text="Configura epochs, batch size y learning rate antes de lanzar el entrenamiento.",
            font=("Segoe UI", 10),
            bg="#0B1120",
            fg="#E5E7EB",
            anchor="w",
        ).pack(side="left", padx=10)

        # CUERPO
        body = tk.Frame(main, bg=COLOR_DARK)
        body.pack(fill="both", expand=True, pady=(16, 0))

        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)

        # -------- COLUMNA IZQUIERDA: HYPER PARAMS -------- #
        left = tk.Frame(body, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        tk.Label(
            left,
            text="Configuración de entrenamiento",
            font=("Segoe UI", 13, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 4))

        tk.Label(
            left,
            text="Ajusta los parámetros según el tamaño del dataset y los recursos disponibles.",
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            wraplength=320,
            justify="left",
        ).pack(fill="x", padx=18, pady=(0, 8))

        # Tarjeta de sliders
        sliders_card = tk.Frame(left, bg=COLOR_CARD)
        sliders_card.pack(fill="x", padx=10, pady=(4, 10))

        # Epochs
        self.crear_slider(
            texto="Epochs",
            desde=10,
            hasta=100,
            variable=self.epochs,
            parent=sliders_card,
            descripcion="Más epochs permiten aprender mejor, pero aumentan el riesgo de sobreajuste.",
        )

        # Batch size
        self.crear_slider(
            texto="Batch size",
            desde=4,
            hasta=32,
            variable=self.batch_size,
            parent=sliders_card,
            descripcion="Con datasets pequeños suele funcionar mejor un batch size reducido.",
        )

        # Learning rate
        frame_lr = tk.Frame(sliders_card, bg=COLOR_CARD)
        frame_lr.pack(fill="x", pady=(8, 4))

        tk.Label(
            frame_lr,
            text="Learning rate (x10⁻⁵)",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(4, 0))

        tk.Label(
            frame_lr,
            text="Valores bajos entrenan más estable; valores altos aprenden rápido pero pueden ser inestables.",
            font=("Segoe UI", 8),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            wraplength=320,
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 4))

        scale_lr = tk.Scale(
            frame_lr,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.lr_scale_var,
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            highlightthickness=0,
            troughcolor="#111827",
        )
        scale_lr.pack(fill="x", padx=8, pady=(0, 4))

        # Botón de entrenamiento
        tk.Button(
            left,
            text="🚀 Iniciar entrenamiento",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_SUCCESS,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self.iniciar_entrenamiento,
        ).pack(fill="x", padx=18, pady=(12, 18))

        # -------- COLUMNA DERECHA: INFO / RECOMENDACIONES -------- #
        right = tk.Frame(body, bg=COLOR_DARK)
        right.grid(row=0, column=1, sticky="nsew")

        # Card principal
        card_info = tk.Frame(right, bg=COLOR_CARD, highlightthickness=1, highlightbackground=COLOR_BORDER)
        card_info.pack(fill="both", expand=True)

        tk.Label(
            card_info,
            text="Recomendaciones de uso",
            font=("Segoe UI", 13, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 4))

        txt = (
            "• Verifica que el directorio de imágenes esté organizado por carpetas (una por estudiante).\n\n"
            "• Con datasets pequeños:\n"
            "   - Usa menos epochs (20–40).\n"
            "   - Usa un batch size pequeño (4–8).\n\n"
            "• Si el modelo sobreajusta:\n"
            "   - Aumenta el dataset con más imágenes.\n"
            "   - Ajusta epochs y reduce el learning rate.\n\n"
            "• El modelo entrenado se guarda como 'modelo_estudiantes.h5' en el directorio del proyecto.\n"
            "• Este modelo se utiliza luego en los módulos de detección en vivo y diagnóstico."
        )
        tk.Label(
            card_info,
            text=txt,
            font=("Segoe UI", 9),
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            justify="left",
            wraplength=520,
        ).pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.root.mainloop()

    # -------- UI HELPER (SOLO PRESENTACIÓN) -------- #

    def crear_slider(self, texto, desde, hasta, variable, parent, descripcion=""):
        frame = tk.Frame(parent, bg=COLOR_CARD)
        frame.pack(fill="x", pady=8)

        tk.Label(
            frame,
            text=texto,
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(4, 0))

        if descripcion:
            tk.Label(
                frame,
                text=descripcion,
                font=("Segoe UI", 8),
                bg=COLOR_CARD,
                fg=COLOR_MUTED,
                wraplength=320,
                justify="left",
            ).pack(fill="x", padx=8, pady=(0, 2))

        scale = tk.Scale(
            frame,
            from_=desde,
            to=hasta,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            highlightthickness=0,
            troughcolor="#111827",
        )
        scale.pack(fill="x", padx=8, pady=(0, 2))

    # -------- LÓGICA ORIGINAL DE ENTRENAMIENTO -------- #

    def iniciar_entrenamiento(self):
        lr = self.lr_scale_var.get() * 0.00001
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()

        # Ventana de progreso
        ventana = VentanaProgreso(self.root, "Entrenamiento", "Preparando...", determinado=True)

        def entrenar():
            # Carga del dataset
            try:
                train_dataset = tf.keras.utils.image_dataset_from_directory(
                    DATA_DIR,
                    validation_split=0.2,
                    subset="training",
                    seed=123,
                    image_size=(128, 128),
                    batch_size=batch_size,
                    shuffle=True,
                )
                val_dataset = tf.keras.utils.image_dataset_from_directory(
                    DATA_DIR,
                    validation_split=0.2,
                    subset="validation",
                    seed=123,
                    image_size=(128, 128),
                    batch_size=batch_size,
                    shuffle=True,
                )

                # Guardar class_names ANTES de normalizar
                class_names = train_dataset.class_names

                # Normalizar datasets
                train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
                val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

            except Exception as e:
                ventana.cerrar()
                messagebox.showerror("Error", f"No se pudo cargar el dataset: {str(e)}")
                return

            num_clases = len(class_names)

            # Construcción CNN MEJORADA
            model = Sequential([
                Conv2D(32, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(),

                Conv2D(64, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(),

                Conv2D(128, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(),

                Conv2D(256, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(),

                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(num_clases, activation='softmax'),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )

            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7),
                CallbackGUI(ventana, epochs),
            ]

            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
            )

            # Guardar modelo
            model.save("modelo_estudiantes.h5")

            # Guardar nombres de clases
            with open('clases.txt', 'w', encoding='utf-8') as f:
                for clase in class_names:
                    f.write(f"{clase}\n")

            ventana.cerrar()
            messagebox.showinfo("Entrenamiento", "Modelo entrenado y guardado exitosamente.")

        threading.Thread(target=entrenar, daemon=True).start()


if __name__ == '__main__':
    EntrenamientoModelo()
