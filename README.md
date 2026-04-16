# KORA

**KORA** es un sistema de asistente inteligente enfocado en la detección de una palabra clave (*wake word*) y actividad de voz (*Voice Activity Detection, VAD*). Su objetivo es funcionar como el módulo de escucha inicial de un asistente más grande, optimizado para dispositivos con recursos limitados y arquitecturas distribuidas.

El sistema captura audio en tiempo real, detecta cuándo una persona habla, almacena el fragmento relevante y lo envía a un servidor externo para el procesamiento de intención y respuesta.

---

# Objetivos del Proyecto

* Detectar una palabra clave en tiempo real.
* Implementar detección de actividad de voz (VAD).
* Reducir el consumo de memoria y CPU.
* Permitir integración con microcontroladores y sistemas embebidos.
* Servir como base para un asistente domótico inteligente.

---

# Características

* Procesamiento de audio en tiempo real.
* Detección eficiente de wake word.
* Buffer de audio optimizado.
* Arquitectura modular.
* Compatible con procesamiento en servidor.
* Escalable para sistemas de automatización y robótica.

---

# Arquitectura del Sistema

El flujo principal del sistema es el siguiente:

1. Captura de audio desde micrófono.
2. Aplicación de VAD para detectar voz.
3. Detección de la palabra clave.
4. Almacenamiento del audio relevante.
5. Envío del audio al servidor para análisis de intención.
6. Recepción de respuesta del servidor.

```
Micrófono
   ↓
Preprocesamiento de audio
   ↓
VAD
   ↓
Wake Word Detection
   ↓
Buffer de audio
   ↓
Servidor de procesamiento
```

---

# Tecnologías Utilizadas

* Python
* Machine Learning
* Procesamiento de audio
* Redes neuronales
* Streaming de audio
* Sistemas embebidos

Opcionalmente:

* TensorFlow / Keras
* PyTorch
* ONNX
* WebSockets o HTTP

---

# Requisitos

Ejemplo de dependencias comunes:

```
Python >= 3.9
numpy
scipy
sounddevice
webrtcvad
librosa
```

Instalación rápida:

```
pip install -r requirements.txt
```

---

# Instalación

Clonar el repositorio:

```
git clone https://github.com/tu-usuario/KORA.git
cd KORA
```

Instalar dependencias:

```
pip install -r requirements.txt
```

---

# Uso

El archivo principal `main.py` se utiliza para ejecutar el entrenamiento de los modelos y realizar pruebas con los modelos de lenguaje.

Ejecutar entrenamiento o pruebas principales:

```
python main.py
```

El sistema puede:

* Entrenar modelos de detección de audio o lenguaje
* Evaluar modelos existentes
* Ejecutar pruebas experimentales de inferencia
* Validar el rendimiento del pipeline de procesamiento

---

# Roadmap

Futuras mejoras planificadas:

* separacion servidor/dispositivo
* Optimización para microcontroladores
* Implementación en ESP32
* Reducción de latencia
* Integración con sistemas domóticos
* Soporte para múltiples wake words
* Interfaz de configuración remota

---

# Estado del Proyecto

En desarrollo activo.

Este proyecto forma parte de un sistema mayor de asistente inteligente modular orientado a automatización, robótica y sistemas embebidos.

---

# Autor

Julio César Salazar Hernández
Estudiante de Ingeniería en Robótica

---

# Contribuciones

Las contribuciones son bienvenidas.

Cualquier persona puede:

* Enviar *pull requests*
* Proponer mejoras en el código
* Optimizar modelos o algoritmos
* Reportar errores
* Extender la arquitectura del sistema

Se busca fomentar un entorno abierto de colaboración técnica y aprendizaje.

---

# Licencia

Este proyecto es de código abierto.

Cualquier persona puede:

* Usar la arquitectura en sus propios proyectos
* Adaptar los modelos a sus necesidades
* Copiar, modificar y redistribuir el código
* Integrar componentes en otros sistemas

Se recomienda mantener el crédito al proyecto original cuando se reutilice la arquitectura o partes significativas del código.

