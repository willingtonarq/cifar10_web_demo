# CIFAR-10: red densa + FastAPI + página web

Este proyecto integra un modelo Keras entrenado con CIFAR-10, una API REST construida con FastAPI y una página web para cargar imágenes y consultar la predicción.

## Estructura del proyecto

```text
CIFAR10_FastAPI_Web_Starter/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── cifar10_dense_model.keras   # Debes copiar aquí tu modelo entrenado
└── cifar10_web_demo/
    └── index.html
```

## Requisitos

- Windows con Python instalado.
- El archivo `cifar10_dense_model.keras` generado durante el entrenamiento.
- Navegador web moderno.

> **Importante:** no necesitas instalar TensorFlow.js ni convertir el modelo a `model.json`. La predicción se ejecuta en Python mediante la API.

## 1. Copiar el modelo

Copia el archivo `cifar10_dense_model.keras` que descargaste de Colab dentro de la carpeta `backend/`.

Verifica que la ruta final sea:

```text
backend/cifar10_dense_model.keras
```

## 2. Crear y activar el entorno virtual del backend

Abre **CMD** (Símbolo del sistema) y cambia a la carpeta `backend` del proyecto. Por ejemplo, si descomprimiste el proyecto en Descargas:

```bat
cd %USERPROFILE%\Downloads\CIFAR10_FastAPI_Web_Starter\backend
```

Si lo descomprimiste en otra ubicación, ajusta la ruta del comando `cd`.

Crea un entorno virtual:

```bat
py -m venv .venv
```

Actívalo:

```bat
.venv\Scripts\activate.bat
```

Cuando esté activo, normalmente verás `(.venv)` al comienzo de la línea de comandos.

Actualiza `pip` e instala las dependencias:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

La instalación de TensorFlow puede tardar varios minutos. Si `pip` informa que no existe una distribución compatible con tu versión de Python, revisa la versión instalada con:

```bat
python --version
```

En ese caso, no cambies paquetes al azar: utiliza una versión de Python compatible con la versión de TensorFlow que vayas a instalar y crea de nuevo el entorno virtual con esa versión.

## 3. Iniciar la API

Desde la misma ventana de CMD, dentro de `backend` y con el entorno `.venv` activado, ejecuta:

```bat
uvicorn main:app --reload
```

Deja esta ventana abierta mientras utilizas la aplicación.

Comprueba estos endpoints en el navegador:

- API: http://127.0.0.1:8000
- Documentación interactiva: http://127.0.0.1:8000/docs
- Estado de la API y del modelo: http://127.0.0.1:8000/health

En `/health`, la respuesta esperada incluye `"status": "ok"` y `"model_loaded": true`.

## 4. Iniciar la página web

Abre **una segunda ventana de CMD**. Entra en la carpeta `cifar10_web_demo`. Si el proyecto está en Descargas, ejecuta:

```bat
cd %USERPROFILE%\Downloads\CIFAR10_FastAPI_Web_Starter\cifar10_web_demo
```

Ajusta la ruta si descomprimiste el proyecto en otra ubicación. Luego inicia un servidor web sencillo:

```bat
py -m http.server 5500
```

Deja esta segunda ventana abierta y visita:

http://127.0.0.1:5500

Selecciona una imagen y pulsa **Clasificar imagen**. La página enviará la imagen a la API y mostrará la clase predicha y las tres mejores probabilidades.

## 5. ¿Cómo funciona?

1. El usuario selecciona una imagen en la página web.
2. JavaScript envía el archivo mediante una petición HTTP `POST` al endpoint `/predict`.
3. FastAPI abre la imagen, la convierte a RGB, la redimensiona a `32 × 32` y normaliza los píxeles al rango `[0, 1]`.
4. El modelo Keras genera las puntuaciones para las 10 clases de CIFAR-10.
5. La API devuelve la clase principal y las tres clases con mayor puntuación.

## 6. Solución de problemas

### La API indica que no encuentra el modelo

Verifica que `cifar10_dense_model.keras` esté dentro de `backend/`, al mismo nivel que `main.py`.

### La página no se conecta con la API

- Confirma que la API esté ejecutándose en `http://127.0.0.1:8000`.
- Abre `http://127.0.0.1:8000/health` y revisa que el modelo esté cargado.
- Mantén abiertas las dos ventanas de CMD: una para FastAPI y otra para el servidor web.

### Error al instalar TensorFlow

Comprueba la versión de Python con `python --version`. TensorFlow debe tener una distribución compatible con la versión de Python y el sistema operativo. No es necesario instalar TensorFlow.js para este proyecto.

## Nota sobre las predicciones

CIFAR-10 contiene imágenes pequeñas de `32 × 32` píxeles. Las fotografías reales pueden clasificarse incorrectamente al reducirse a ese tamaño. La puntuación devuelta por el modelo no es una garantía de acierto ni necesariamente una probabilidad calibrada.
