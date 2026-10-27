# ALIGN LAB

Laboratorio interactivo de **registro y alineación geométrica de imágenes**. Cargas una imagen de entrada y una plantilla de referencia; la app detecta puntos clave (SIFT u ORB), los empareja con el *Lowe ratio test*, estima una homografía con RANSAC, transforma la entrada al sistema de coordenadas de la plantilla y te muestra métricas calculadas de verdad.

## Qué hace

- Carga y validación (PNG, JPG, TIFF; gris, BGR y BGRA) con límites de tamaño (25 MB) y resolución (25 MP).
- Modo **Automático** y modo **Experto** (detector, máximo de características, Lowe ratio, umbral RANSAC, % de matches).
- Pestañas: original, alineación, superposición con opacidad ajustable, correspondencias, inliers/outliers de RANSAC y diferencias (con mapa térmico y máscara de cobertura).
- Métricas: keypoints, matches, inliers/outliers, tasa de inliers, error de reproyección sobre inliers, cobertura espacial de inliers, tiempo.
- Comparación SIFT/ORB con barra de progreso; un fallo se muestra como fallo, no como ceros.
- Descargas: `aligned.png`, `matches.png`, `overlay.png`, `differences.png`, `metrics.json`, `homography.json` y ZIP. Corresponden a la ejecución mostrada en pantalla.

## Estructura

```text
align-lab/
├── app.py                 # Interfaz Streamlit
├── requirements.txt
├── .streamlit/config.toml # Tema oscuro
├── src/
│   ├── detectors.py       # SIFT/ORB y disponibilidad
│   ├── matching.py        # Matcher (derivado del detector) y ratio test
│   ├── registration.py    # Homografía, validaciones, warp y máscara de cobertura
│   ├── metrics.py         # Métricas y tiempos
│   ├── visualization.py   # Overlay, matches, diferencias
│   ├── export.py          # PNG, JSON y ZIP en memoria
│   └── pipeline.py        # Decodificación segura y flujo completo (sin Streamlit)
├── tests/                 # Pruebas con imágenes sintéticas
└── samples/README.md
```

`pipeline.py` es un añadido a la arquitectura pedida: mantiene la lógica fuera de `app.py` y permite probarla sin Streamlit.

## Ejecución local (Windows + VS Code)

```powershell
mkdir align-lab
cd align-lab          # (copia aquí los archivos del proyecto)

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Se abre en `http://localhost:8501`.

**Si PowerShell bloquea la activación** del entorno, no hace falta cambiar políticas globales: usa el intérprete del entorno directamente, sin activarlo.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**VS Code:** `Ctrl+Shift+P` → *Python: Select Interpreter* → elige `.\.venv\Scripts\python.exe`.

## Pruebas

```powershell
python -m pytest -q
```

Usan imágenes sintéticas y comprueban, entre otras cosas, que la homografía se recupera a partir de puntos y de una transformación conocidos, que los casos degenerados se rechazan y que la exportación contiene los valores reales.

## Cómo interpretar las métricas

- **Tasa de inliers** = inliers / matches usados. No es la probabilidad de que la alineación sea correcta.
- **Error de reproyección** = distancia media (px de la plantilla) entre `H·p_entrada` y `p_plantilla`, solo sobre inliers.
- **Cobertura espacial** = área del cierre convexo de los inliers / área de la plantilla. Si es baja, los puntos están concentrados y la homografía puede ser poco fiable lejos de ellos.
- **Diferencias**: se calculan solo donde el warp tiene píxeles válidos; la iluminación, el color y la interpolación también las afectan.

## Decisiones técnicas

- **Matcher sincronizado con el detector:** se usa el nombre del detector *efectivo* (`resolve_detector_name`) para elegir el matcher; si SIFT no existe y se cae a ORB, el matcher es Hamming. No hay booleanos intermedios.
- **SIFT** usa `BFMatcher` con L2 (exacto y sin parámetros frágiles); **ORB** usa `BFMatcher` con Hamming.
- **Matches:** por defecto se usan *todos* los que pasan el ratio test (el script original descartaba el 85 %). En modo Experto puedes volver a filtrar por porcentaje.
- **Dependencia:** `opencv-contrib-python-headless` (SIFT está en el núcleo de OpenCV ≥ 4.4, pero se usa *contrib* según la especificación; la variante *headless* evita problemas de librerías gráficas en servidores). No instales a la vez otra variante de OpenCV.
- Se eliminaron `Pillow` y `matplotlib` de las dependencias porque no se usan.

## Limitaciones

- La homografía modela escenas planas o vistas con perspectiva; no corrige deformaciones no rígidas ni escenas 3D con mucho relieve.
- Poca textura, patrones repetitivos, desenfoque o grandes cambios de iluminación degradan los resultados.
- Muchos inliers no garantizan una buena alineación.
- Los tiempos dependen del hardware; el hosting gratuito tiene CPU y memoria limitadas.
- Aún no está implementado el procesamiento por lotes (la arquitectura lo permite: `pipeline.align_images` y `compare_detectors` trabajan sobre una pareja de arrays).

## Privacidad

Las imágenes se procesan **en memoria**, en el servidor donde se ejecuta Streamlit: tu equipo en local; el servidor de la app en la versión publicada (no tu navegador). No se escriben a disco ni se envían a servicios externos, y la telemetría de Streamlit está desactivada en `config.toml`. Usa "Limpiar resultados" para borrar de la sesión las imágenes procesadas.

## Publicación en Streamlit Community Cloud

1. Crea un repositorio en GitHub llamado `align-lab`.
2. Añade `app.py`, `src/`, `tests/`, `requirements.txt`, `.streamlit/config.toml`, `README.md` y `.gitignore`. **No subas** imágenes privadas, tokens ni archivos grandes (el `.gitignore` excluye imágenes en `samples/`).
3. Sube el código:
   ```powershell
   git init
   git add .
   git commit -m "ALIGN LAB"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/align-lab.git
   git push -u origin main
   ```
4. Entra en [share.streamlit.io](https://share.streamlit.io), pulsa **Create app**, elige el repositorio, la rama `main` y el archivo `app.py`.
5. En **Advanced settings** selecciona Python 3.11 (o la versión que ofrezca y sea compatible con `opencv-contrib-python-headless`) y confirma el despliegue.
6. Prueba la URL pública con imágenes de ejemplo y con archivos inválidos.
7. Para actualizar, haz `git push` a `main`: la app se redespliega sola.

Esta guía describe el procedimiento; la app **no está desplegada** hasta que lo hagas tú.
