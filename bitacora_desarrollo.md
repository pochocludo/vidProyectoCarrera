# Bitácora de Desarrollo: Sistema Embebido de Identificación Varietal de Vid

> **Proyecto:** Identificación automática de variedades de vid (*Vitis vinifera L.*) mediante redes neuronales convolucionales ligeras en plataformas embebidas.  
> **Institución:** Universidad Autónoma de Baja California (UABC) — Facultad de Ingeniería, Mexicali.  
> **Equipo de Desarrollo:** Jesús Ramón Ruiz Soto, Sergio Josué López García.  
> **Director de Proyecto:** Dr. Adolfo Heriberto Ruelas Puente.  
> **Ubicación de Código y Datos:** `documents/python/`

---

## Control de Versiones y Estado del Proyecto

* **Fase Actual:** Etapa 1 — Integración, Curaduría y Preparación de Datos.
* **Plataforma Objetivo:** Raspberry Pi 5 (2 GB RAM) + Raspberry Pi Camera Module 3 (12 MP, Autofoco).
* **Entorno de Desarrollo:** Python 3.14 / 64-bit sobre Windows (estación de trabajo previa a migración embebida).

---

## Registro Cronológico de Actividades

### [2026-10-09] — Entrada 01: Análisis y Desglose del Documento Base del Proyecto
* **Etapa Metodológica:** Fase Preliminar / Planificación.
* **Componente Afectado:** Documentación y Especificación Técnica.
* **Descripción de la Actividad:**
  - Se realizó una revisión exhaustiva del archivo de avance de tesis [capitulo3_avance.pdf](file:///C:/Users/jossu/OneDrive/Documentos/python/docs/capitulo3_avance.pdf).
  - Se extrajeron y formalizaron las 4 hipótesis cuantificables ($H_{1a}$, $H_{1b}$, $H_{1c}$, $H_{1d}$) y los requerimientos funcionales/no funcionales (RF-01 a RF-06, RD-01 a RD-05, RNF-01 a RNF-05).
  - Se identificaron los datasets clave en la literatura revisada: GLVD (Shah et al., 2026), Kaggle Grapevine Leaves (Vlah, 2021; Koklu et al., 2022b), UTAD (Carneiro, 2024) y el conjunto local ciego de Baja California (López García, 2025).
* **Decisiones de Diseño Tomadas:**
  1. Mantener el conjunto local de *Misión*, *Syrah* y *Tempranillo* estrictamente aislado de todo entrenamiento y calibración para garantizar la validez externa.
  2. Implementar un mecanismo de rechazo basado en umbral de confianza softmax ($\tau$) para clasificar la variedad *Misión* como "no reconocida" al tratarse de un problema de conjunto abierto.
* **Estado:** ✅ Concluido.

---

### [2026-10-09] — Entrada 02: Generación de la Guía Metodológica Integral
* **Etapa Metodológica:** Planificación y Arquitectura de Software.
* **Componente Afectado:** Archivo [guia_desarrollo_red_neuronal.md](file:///C:/Users/jossu/OneDrive/Documentos/python/guia_desarrollo_red_neuronal.md).
* **Descripción de la Actividad:**
  - Redacción y publicación de la guía técnica que condensa los protocolos de preprocesamiento ($224 \times 224$ con normalización ImageNet), aumento de datos estricto solo en partición de entrenamiento (prevención de fuga de datos), esquema de transferencia en dos fases y cuantización posterior al entrenamiento (PTQ INT8).
  - Definición del cálculo de intervalos de confianza de Wilson al 95% para muestras externas reducidas.
* **Estado:** ✅ Concluido.

---

### [2026-10-09] — Entrada 03: Estructuración del Entorno y Creación de Directorios
* **Etapa Metodológica:** Etapa 1 — Adquisición de Datos.
* **Componente Afectado:** Sistema de Archivos (`documents/python/datasets/`).
* **Descripción de la Actividad:**
  - Se creó el directorio de almacenamiento `datasets/` dentro del espacio de trabajo `python/` para alojar los repositorios crudos y procesados.
* **Estado:** ✅ Concluido.

---

### [2026-10-09] — Entrada 04: Localización y Automatización de Descarga de GLVD (Zenodo)
* **Etapa Metodológica:** Etapa 1 — Adquisición de Datos.
* **Componente Afectado:** Script [download_datasets.py](file:///C:/Users/jossu/OneDrive/Documentos/python/download_datasets.py).
* **Descripción de la Actividad:**
  - Se rastreó el registro oficial del dataset GLVD en Zenodo (DOI: `10.5281/zenodo.18937397`).
  - Se desarrolló e iteró el script de descarga directa desde el endpoint REST de Zenodo (`https://zenodo.org/api/records/18937397/files/.../content`).
  - **Incidencia detectada:** Los servidores de Zenodo limitan conexiones prolongadas, provocando cierres inesperados de socket alrededor de ~45 MB.
  - **Solución implementada:** Se integró soporte para *HTTP Range Requests* (encabezado `Range: bytes=X-`) con reanudación automática de descargas parciales, reintentos exponenciales y descompresión automática tras verificar integridad.
* **Resultado:** Descarga continua estable de los 1,053.1 MB del dataset completo hacia `datasets/GLVD.zip` con recuperación automática por cortes de red.
* **Estado:** ✅ Concluido.

---

### [2026-10-09] — Entrada 05: Extracción y Verificación de Integridad de GLVD
* **Etapa Metodológica:** Etapa 1 — Integración y Curaduría de Datos.
* **Componente Afectado:** `datasets/GLVD/Grapevine Leaf Variety & Disease Dataset (GLVD)/Leaves/`
* **Descripción de la Actividad:**
  - Se ejecutó el script de verificación `verify_dataset.py` sobre los datos descomprimidos.
  - Se confirmó la estructura de particiones `train`, `val` y `test` para las **16 clases de hojas**:
    - **Train:** 1,359 imágenes (incluye 109 de *Syrah* y 60 de *Tempranillo*).
    - **Validation:** 80 imágenes (5 imágenes por clase, balanceadas).
    - **Test interno:** 80 imágenes (5 imágenes por clase, balanceadas).
    - **Total foliar:** 1,519 imágenes etiquetadas y ordenadas por cultivar.
  - Se constató la presencia de las dos clases de interés botánico regional presentes en conjuntos públicos (*Syrah* y *Tempranillo*).
* **Estado:** ✅ Concluido.

---

### [2026-10-09] — Entrada 06: Implementación de Pasos 2 y 3 (Dataset Loader, Modelos y Pipeline de Entrenamiento)
* **Etapa Metodológica:** Etapa 2 — Entrenamiento y Selección de Arquitectura.
* **Componentes Creados:**
  - `requirements.txt`: Definición de dependencias base (`tensorflow`, `keras`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`).
  - `src/dataset.py`: Módulo de carga de particiones `train`, `val`, `test` con `data_augmentation` activo exclusivamente en `train`, interpolación bicúbica a $224 \times 224$ y cálculo de pesos inversos por clase para mitigar desbalance.
  - `src/models.py`: Constructor modular de arquitecturas (`MobileNetV3-Small`, `MobileNetV2`, `EfficientNet-B0`, `ResNet-50`) con pesos de ImageNet, normalización integrada por arquitectura y configuraciones para Fase 1 (base congelada, $lr=10^{-3}$) y Fase 2 (fine-tuning gradual, $lr=10^{-5}$).
  - `train.py`: Script principal ejecutable desde el entorno de Antigravity, con reporte de métricas por clase, evaluación automática de la hipótesis $H_{1a}$ ($\text{Top-1} \ge 90\%$, $\text{Macro } F_1 \ge 0.88$) y guardado del modelo óptimo en `.keras`.
* **Estado:** ✅ Concluido.

---

## Registro de Incidencias Técnicas y Soluciones

| ID | Fecha | Problema Detectado | Causa Raíz | Solución Aplicada |
| :---: | :---: | :--- | :--- | :--- |
| **INC-01** | 2026-10-09 | Error HTTP 403 Forbidden al descargar GLVD. | El enlace web HTML directo de Zenodo exige cabeceras de sesión de navegador. | Se utilizó el endpoint oficial de la REST API (`/api/records/.../content`) con User-Agent emulado. |
| **INC-02** | 2026-10-09 | Descarga truncada a los 44.7 MB y error `BadZipFile`. | La conexión TCP se interrumpió y el lector estándar interpretó el fin de flujo como éxito. | Se implementó descarga con verificación de tamaño total y reanudación incremental mediante `Range`. |

---

## Próximos Pasos en el Roadmap Inmediato

1. **Verificación y Extracción de GLVD:**
   - Comprobar que las 16 clases de cultivares estén presentes en `datasets/GLVD/Leaves/` con las particiones `train` (1309), `val` (80) y `test` (80).
2. **Script de Homologación y Deduplicación (`dataset_curation.py`):**
   - Normalizar nomenclaturas y verificar si existen imágenes duplicadas o casi duplicadas mediante hash perceptual (`imagehash`).
3. **Módulo de DataLoaders (`dataset.py`):**
   - Construir el pipeline de lectura de imágenes con redimensionamiento $224 \times 224$ e ImageNet transforms.
4. **Construcción del Pipeline de Modelos (`models.py`):**
   - Cargar arquitecturas `MobileNetV3-Small`, `MobileNetV2`, `EfficientNet-B0` y `ResNet-50` con pesos ImageNet y cabezal de clasificación multiclase.
