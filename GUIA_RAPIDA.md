# 🚀 Guía Rápida - Mistral Fine-tuning

## ⚡ Inicio Rápido

### 1. Instalación (Una sola vez)
```bash
cd "Instalar Requesitos"
python InstalarTodo.py
```

### 2. Verificación
```bash
python VerificarTodo.py
```

### 3. Ejecutar Fine-tuning
```bash
cd "../RealizarFineTuning"
python mistral_finetuning_final.py
```

## 📁 Estructura del Proyecto

```
Fine/
├── 📋 README.md                     # Documentación completa
├── 📋 GUIA_RAPIDA.md               # Esta guía
├── 📋 requirements.txt             # Dependencias
├── 🔒 .gitignore                   # Exclusiones Git
│
├── 🔧 Instalar Requesitos/         # 🔧 INSTALACIÓN
│   ├── InstalarTodo.py            # ← Ejecutar primero
│   └── VerificarTodo.py           # ← Luego verificar
│
├── 🗂️ Dataset de Miramar/          # 🗂️ TUS DATOS
│   └── *.jsonl                   # ← Tu dataset aquí
│
├── 🚀 RealizarFineTuning/          # 🚀 ENTRENAMIENTO
│   └── mistral_finetuning_final.py # ← Script principal
│
└── 📤 outputs/                     # 📤 RESULTADOS
    └── mistral_finetuned_*/       # ← Modelos entrenados
```

## 🎯 Configuraciones de Dataset

| Opción | Descripción | Cuándo usar |
|--------|-------------|-------------|
| **1** 🚛 | Solo Miramar | Máxima especialización |
| **2** 🌍 | Miramar + Base | **RECOMENDADO** - Equilibrio |
| **3** 📚 | Solo Base | Sin especialización |

## ⚙️ Requisitos Mínimos

- **GPU**: NVIDIA 8GB+ (RTX 3070, 4060 Ti, etc.)
- **RAM**: 16GB+
- **Espacio**: 50GB libres
- **CUDA**: 11.8+

## 🔧 Solución Rápida de Problemas

### Error: "CUDA out of memory"
```python
# En mistral_finetuning_final.py, línea ~275:
per_device_train_batch_size = 1  # Cambiar de 2 a 1
```

### Error: "Dataset no encontrado"
```bash
# Verificar que esté en la ubicación correcta:
ls "Dataset de Miramar/"
```

### Error: "Unsloth not found"
```bash
cd "Instalar Requesitos"
python InstalarTodo.py
```

## ⏱️ Tiempos Estimados

| GPU | Tiempo Fine-tuning | Batch Size |
|-----|-------------------|------------|
| RTX 3070 | 15-20 min | 1-2 |
| RTX 4070 Ti | 8-12 min | 2-4 |
| RTX 4090 | 3-5 min | 8-16 |

## 📤 Resultados

El modelo se guarda en:
```
outputs/mistral_finetuned_[config]/
├── adapter_model.safetensors  # Modelo LoRA
├── tokenizer.json            # Tokenizador
└── unsloth.Q4_K_M.gguf      # Formato optimizado
```

## 🆘 Soporte

- 🐛 **Problemas**: Ejecuta `python VerificarTodo.py`
- 📖 **Documentación completa**: Lee `README.md`
- 🔄 **Reinstalar**: Ejecuta `python InstalarTodo.py`

---
*¿Todo funcionando? ¡Dale una ⭐ al proyecto!*
