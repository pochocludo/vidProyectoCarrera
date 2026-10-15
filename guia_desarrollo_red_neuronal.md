# Guía Integral para el Desarrollo de la Red Neuronal de Identificación Varietal de Vid

> **Proyecto de Carrera (UABC - Facultad de Ingeniería, Mexicali, B.C.)**  
> **Autores:** Jesús Ramón Ruiz Soto y Sergio Josué López García  
> **Asesor:** Dr. Adolfo Heriberto Ruelas Puente  
> **Documento Base:** *Capítulo 3: Diseño y desarrollo de la solución* (`capitulo3_avance.pdf`)

---

## 1. Visión General y Alcance del Proyecto

El objetivo central es desarrollar e implementar un **sistema embebido autónomo** basado en redes neuronales convolucionales (CNN) ligeras para la **identificación varietal de vid (*Vitis vinifera L.*)** mediante el análisis fotográfico de hojas adultas individuales.

### 1.1. Principales Restricciones y Delimitaciones
* **Sujeto de análisis:** Hojas adultas completas y extendidas, fotografiadas individualmente. No contempla racimos, sarmientos, brotes jóvenes ni diagnóstico patológico (enfermedades).
* **Alcance de predicción:** Limitado estrictamente a las variedades presentes en los conjuntos de entrenamiento públicos.
* **Manejo de conjunto abierto (Open-Set Recognition):** La variedad histórica local **Misión** (*Listán Prieto*) no existe en conjuntos públicos abiertos. El sistema no la identificará positivamente, sino que la **rechazará como "variedad no reconocida"** mediante un umbral de confianza calibrado ($\tau$).
* **Entorno operativo:** Sin dependencia de conexión a internet (*offline*), optimizado para operar en campo (viñedo) sobre una computadora de placa única **Raspberry Pi 5** con cámara de enfoque automático (Raspberry Pi Camera Module 3).

---

## 2. Marco Metodológico: Hipótesis y Criterios de Aceptación

El diseño experimental se rige por cuatro hipótesis específicas con umbrales cuantitativos que determinan la aceptación de la hipótesis general ($H_1$):

| Código | Hipótesis Específica | Métrica y Criterio de Aceptación | Conjunto de Evaluación |
| :--- | :--- | :--- | :--- |
| **$H_{1a}$** | Desempeño interno en dominio público | $\text{Exactitud top-1} \ge 90\%$ y $\text{Macro } F_1 \ge 0.88$ | Partición de prueba interna de datos públicos |
| **$H_{1b}$** | Generalización externa en variedades conocidas | $\text{Exactitud top-1} \ge 70\%$ | Conjunto local UABC (*Syrah* y *Tempranillo*) |
| **$H_{1c}$** | Rechazo efectivo de variedades no vistas | $\text{Tasa de rechazo} \ge 70\%$ | Conjunto local UABC (*Misión*) |
| **$H_{1d}$** | Viabilidad y eficiencia en hardware embebido | $\text{Latencia inferencia} \le 500\text{ ms}$ y $\Delta\text{Exactitud por INT8} \le 2\%$ | Raspberry Pi 5 (modelo cuantizado INT8) |

> **Nota metodológica (Intervalos de Wilson):** Dado el tamaño reducido del conjunto local de prueba de Baja California, todas las métricas sobre el conjunto externo deben reportarse con **intervalos de confianza al 95% calculados mediante el método de Wilson**:
> $$IC = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z \approx 1.96)$$

---

## 3. Flujo Metodológico por Etapas

El proyecto se estructura en siete etapas secuenciales con entregables definidos:

```
[1. Curaduría y Datos] ──► [2. Entrenamiento y Selección] ──► [3. Optimización INT8]
                                                                     │
[6. Validación Externa] ◄── [5. Implementación Embebida] ◄── [4. Selección Plataforma]
        │
[7. Contraste de Hipótesis]
```

---

## 4. Etapa 1: Integración y Curaduría de Datos

### 4.1. Fuentes de Datos
1. **GLVD (Grapevine Leaf Variety & Disease Dataset - Shah et al., 2026):**  
   Base principal. Reúne 1,469 imágenes de hojas distribuidas en 16 cultivares con particiones predefinidas:
   * *Ak, Ala Idris, Buzgulu, Dimnit, Nazli* (provenientes de Koklu et al.).
   * *Auxerrois, Cabernet Franc, Cabernet Sauvignon, Chardonnay, Merlot, Muller Thurgau, Pinot Noir, Riesling, Sauvignon Blanc, Syrah, Tempranillo* (provenientes de Vlah).
   * Particiones: 1,309 entrenamiento, 80 validación (5 por clase), 80 prueba (5 por clase).
2. **UTAD Grapevine Varieties Dataset 2023 (Carneiro, 2024):**  
   Fuente complementaria para enriquecer variabilidad en condiciones de iluminación y fondo (opcional según requerimiento de volumen).
3. **Conjunto Local UABC / Viñas Pasini (López García, 2025):**  
   Muestras de hojas colectadas en San Antonio de las Minas / Valle de Guadalupe (*Syrah*, *Tempranillo* y *Misión*). **Uso exclusivo en Etapa 6 (prueba ciega).**

### 4.2. Homologación de Sinónimos Varietales (VIVC)
Para evitar tratar la misma variedad botánica bajo nombres distintos:
* `Shiraz` $\longrightarrow$ `Syrah`
* `Tinta Roriz` / `Aragonez` $\longrightarrow$ `Tempranillo`
* `Listán Prieto` / `País` / `Criolla Chica` $\longrightarrow$ `Misión`

### 4.3. Deduplicación por Hash Perceptual (`imagehash`)
Para evitar la fuga de datos (*data leakage*) entre entrenamiento y evaluación:
* Calcular el hash perceptual (dHash o pHash de 64 bits) para todas las imágenes.
* Detectar pares con distancia de Hamming $\le 4$.
* Si existen duplicados compartidos entre fuentes, asegurar que no queden divididos entre el conjunto de entrenamiento y el de prueba.

---

## 5. Etapa 2: Entrenamiento y Selección de Arquitectura

### 5.1. Arquitecturas Candidatas
Se evaluarán cuatro modelos preentrenados en ImageNet:
1. **MobileNetV3-Small (1.0x):** Enfoque ultra ligero (~2.5 M parámetros, 56 M MAdds).
2. **MobileNetV2 (1.0x):** Modelo consolidado en visión móvil (~3.4 M parámetros, 300 M MAdds).
3. **EfficientNet-B0:** Escalado compuesto de alta exactitud (~5.3 M parámetros, 0.39 GFLOPs).
4. **ResNet-50:** Línea base de alta capacidad (~26 M parámetros, 4.1 GFLOPs).

### 5.2. Preprocesamiento Estricto
* **Resolución:** $224 \times 224$ píxeles, interpolación bicúbica.
* **Normalización ImageNet:**
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
* **Regla de oro:** Los parámetros de redimensionamiento y normalización deben ser idénticos durante el entrenamiento, la cuantización y la inferencia en el Raspberry Pi.

### 5.3. Aumento de Datos (Data Augmentation)
* **Condición crítica:** Aplicar aumento de datos **únicamente** en la partición de entrenamiento en tiempo de ejecución (online).
* **Transformaciones permitidas:**
  - Rotación aleatoria ($\pm 30^\circ$).
  - Volteo horizontal y vertical aleatorio.
  - Recorte aleatorio escalado (*Random Resized Crop* escala $[0.8, 1.0]$).
  - Jitter de color leve (brillo $\pm 0.15$, contraste $\pm 0.15$, saturación $\pm 0.1$).

### 5.4. Protocolo de Entrenamiento en Dos Fases
1. **Fase 1 (Ajuste del clasificador):**
   * Base convolucional congelada.
   * Optimizador: `Adam`, learning rate $\eta = 10^{-3}$.
   * Función de pérdida: `Categorical Crossentropy` ponderada por clase para corregir desbalance:
     $$w_k = \frac{N}{K \cdot N_k}$$
   * Épocas: 10 a 15 hasta estabilización.
2. **Fase 2 (Fine-tuning gradual):**
   * Descongelar el último tercio de bloques convolucionales de la red.
   * Optimizador: `Adam`, learning rate reducido $\eta = 10^{-5}$ con cosine decay.
   * Early stopping: Paciencia de 10 épocas monitoreando el **Macro F1-score** en el conjunto de validación.

### 5.5. Evaluación Multisemilla
Entrenar cada una de las 4 arquitecturas con **3 semillas aleatorias distintas** (ej. 42, 123, 999). Reportar media y desviación estándar de:
* Exactitud Top-1 (Accuracy)
* Macro Precision, Macro Recall y Macro F1
* Matriz de confusión
* Criterio de selección: La arquitectura con menor conteo de parámetros y MACs que cumpla $H_{1a}$.

---

## 6. Etapa 3: Calibración de Umbral y Optimización INT8

### 6.1. Calibración del Umbral de Rechazo ($\tau$)
Para clasificar en conjunto abierto:
$$\hat{y} = \begin{cases} \arg\max_k p_k & \text{si } \max_k p_k \ge \tau \\ \text{"Variedad no reconocida"} & \text{en otro caso} \end{cases}$$

* **Procedimiento:**
  1. Pasar la partición de **validación** interna por el modelo entrenado.
  2. Filtrar únicamente las predicciones correctas.
  3. Fijar $\tau$ en el percentil 5 de las confianzas máximas de los aciertos (de modo que conserve el 95% de las predicciones correctas).
  4. **Prohibición metodológica:** El umbral $\tau$ debe quedar congelado **antes** de evaluar el conjunto local de Baja California.

### 6.2. Cuantización Posterior al Entrenamiento (PTQ INT8)
* Convertir el modelo a formato **TensorFlow Lite (`.tflite`)** con cuantización completa de pesos y activaciones a enteros de 8 bits (`int8`).
* Emplear un subconjunto representativo de ~100 imágenes del conjunto de entrenamiento para la calibración del rango dinámico.
* Evaluar la pérdida de exactitud: verificar que $\text{Exactitud}_{\text{FP32}} - \text{Exactitud}_{\text{INT8}} \le 2.0\%$.

---

## 7. Etapas 4 y 5: Implementación en Plataforma Embebida

### 7.1. Especificación del Hardware
* **Cómputo:** Raspberry Pi 5 (2 GB RAM, Quad-Core Cortex-A76 @ 2.4 GHz).
* **Adquisición:** Raspberry Pi Camera Module 3 (Sensor Sony IMX708 de 12 MP con enfoque automático por PDAF, interfaz MIPI CSI-2).
* **Sistema Operativo:** Raspberry Pi OS Lite (64-bit, sin entorno gráfico para no alterar la latencia).
* **Enfriamiento:** Disipador activo oficial Raspberry Pi (cooler con control PWM para evitar *thermal throttling*).

### 7.2. Arquitectura de Software Modular

```text
python/
├── config.py         # Configuración central (rutas, umbral tau, parámetros cámara, clases)
├── captura.py        # Módulo libcamera / Picamera2 con autofoco y balance fijo
├── preproceso.py     # Redimensionamiento 224x224, normalización y formato de tensor
├── inferencia.py     # Carga de TFLite runtime e inferencia INT8
├── decision.py       # Lógica del umbral tau ("variedad" vs "no reconocida")
├── registro.py       # Registro CSV de logs (timestamp, predicción, confianza, tiempos por etapa)
├── main_cli.py       # Interfaz por línea de comandos para benchmarks y pruebas
└── app_web.py        # Interfaz web local ligera para consulta desde smartphone en viñedo
```

### 7.3. Presupuesto de Latencia Extremo a Extremo
$$\text{Latencia Total} = t_{\text{captura}} + t_{\text{preproceso}} + t_{\text{inferencia}} + t_{\text{decisión/log}}$$
* Captura y autofocus: $150 - 250\text{ ms}$
* Preprocesamiento: $10 - 20\text{ ms}$
* Inferencia INT8 (MobileNetV3 en Pi 5): $\approx 8 - 15\text{ ms}$
* Decisión y registro: $< 5\text{ ms}$
* **Presupuesto esperado:** $\approx 180 - 290\text{ ms} \ll 500\text{ ms}$ (Cumple $H_{1d}$).

---

## 8. Etapas 6 y 7: Validación Externa y Contraste de Hipótesis

1. **Prueba ciega local:**
   * Evaluar imágenes de *Syrah* y *Tempranillo* locales: Contrastar si $\text{Exactitud} \ge 70\%$ ($H_{1b}$).
   * Evaluar imágenes de *Misión* locales: Contrastar si $\text{Tasa de rechazo} \ge 70\%$ ($H_{1c}$).
2. **Medición de hardware:**
   * Medir latencia de inferencia sobre 100 repeticiones consecutivas (descartando primeras 5 de warm-up): calcular media y percentil 95 ($P_{95}$).
   * Monitorear consumo de RAM y energía durante la prueba de carga.
3. **Análisis de fallos:**
   * Elaborar matriz de confusión desglosada y análisis de transferibilidad entre dominios.
