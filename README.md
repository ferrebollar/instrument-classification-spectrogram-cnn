# Musical Instrument Family Classification using CNN

Clasificación de familias de instrumentos musicales mediante una red neuronal convolucional (CNN) aplicada sobre espectrogramas Mel para la clasificación de familias instrumentales a partir de audio.

El modelo trabaja sobre representaciones tiempo–frecuencia mediante espectrogramas Mel y clasifica cuatro familias:

- Brass
- Strings
- Winds
- Keyboards

Además del conjunto principal de entrenamiento, se incluyen validaciones externas para analizar la capacidad de generalización del modelo.

## Datasets utilizados

### TinySOL (entrenamiento)

Dataset principal utilizado para entrenar el modelo.

Descarga:

https://zenodo.org/records/3685367

Estructura esperada:

```text
TINYSOL/
├── Brass/
│   ├── Horn/
│   ├── Trombone/
│   ├── Trumpet/
│   └── Bass_Tuba/
│
├── Strings/
│   ├── Violin/
│   ├── Viola/
│   ├── Violoncello/
│   └── Contrabass/
│
├── Winds/
│   ├── Flute/
│   ├── Oboe/
│   ├── Clarinet/
│   ├── Bassoon/
│   └── Saxophone/
│
└── Keyboards/
    └── Accordion/
```

La estructura de carpetas se utiliza para extraer automáticamente las etiquetas de familia e instrumento.

---

### Philharmonia Dataset (validación externa)

Descarga:

https://philharmonia.co.uk/resources/sound-samples/

Crear la carpeta:

```text
Validacion/
├── Brass/
├── Strings/
└── Winds/
```

---

### IRMAS (validación externa)

Descarga:

https://www.upf.edu/web/mtg/irmas

Crear la carpeta:

```text
Validacion2_procesada/
├── Brass/
├── Strings/
├── Winds/
└── Keyboards/
```

---

### NSynth (validación externa)

Descarga:

https://magenta.withgoogle.com/datasets/nsynth

Crear la carpeta:

```text
Validacion_NSynth_v2/
├── Brass/
├── Strings/
└── Winds/
```

---

## Configuración

Modificar la ruta principal del dataset en el notebook:

```python
RUTA = "/content/drive/MyDrive/TINYSOL"
```

Los datasets externos deben colocarse respetando exactamente los nombres de carpeta indicados.

## Dependencias

Instalar las librerías necesarias:

```bash
pip install tensorflow librosa pandas numpy matplotlib seaborn scikit-learn
```

## Arquitectura del modelo

- Entrada: espectrogramas Mel `96×96×1`
- CNN con tres bloques convolucionales (`16 → 32 → 64`)
- MaxPooling `2×2`
- Capa densa de `64` neuronas
- Dropout `0.3`
- Optimizador: Adam
- Función de pérdida: Sparse Categorical Crossentropy

## Evaluación

El modelo se evalúa mediante:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Análisis cualitativo de ejemplos correctos y errores
- Validación externa sobre datasets independientes
