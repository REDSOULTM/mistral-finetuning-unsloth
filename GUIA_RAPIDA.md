# 🚀 Guía rápida para usar este proyecto (sin vueltas)

---

## ⚡ ¿Qué hago primero?

1. **Instala todo lo necesario** (solo una vez):
   ```bash
   cd "Instalar Requesitos"
   python InstalarTodo.py
   ```
2. **Verifica que no falte nada:**
   ```bash
   python VerificarTodo.py
   ```
3. **Entrena tu modelo:**
   ```bash
   cd "../RealizarFineTuning"
   python mistral_finetuning_final.py
   ```

---

## 📁 ¿Cómo está armado esto?

```
Fine/
├── README.md                # Documentación completa
├── GUIA_RAPIDA.md           # Esta guía
├── requirements.txt         # Lo que hay que instalar
├── .gitignore               # Lo que NO se sube a GitHub
│
├── Instalar Requesitos/     # Scripts para instalar y verificar
│   ├── InstalarTodo.py      # Instalador automático
│   └── VerificarTodo.py     # Verificador rápido
│
├── Dataset de Miramar/      # Tu dataset personalizado (jsonl)
│
├── RealizarFineTuning/      # Script principal para entrenar
│   └── mistral_finetuning_final.py
│
└── outputs/                 # Modelos entrenados (se crean solos)
```

---

## 🎯 ¿Qué opciones de dataset tengo?

| Opción | ¿Qué hace? | ¿Cuándo usar? |
|--------|------------|---------------|
| 1 🚛   | Solo Miramar | Si querés que el modelo sea experto SOLO en tu tema |
| 2 🌍   | Miramar + Base | **La mejor**: mezcla tu info con conocimiento general |
| 3 📚   | Solo Base | Si querés un modelo más genérico, sin especialización |

---

## ⚙️ ¿Qué necesito en mi compu?

- **GPU NVIDIA**: 8GB o más (ej: RTX 3070, 4060 Ti, 4070, etc.)
- **RAM**: 16GB mínimo
- **Espacio libre**: 50GB
- **CUDA**: 11.8+

---

## 🔧 Problemas comunes y cómo los arreglo

- **"CUDA out of memory"**
  - Solución: Abrí `mistral_finetuning_final.py`, buscá la línea del batch size y poné:
    ```python
    per_device_train_batch_size = 1
    ```

- **"Dataset no encontrado"**
  - Solución: Fijate que el archivo esté en `Dataset de Miramar/`:
    ```bash
    ls "Dataset de Miramar/"
    ```

- **"Unsloth not found"**
  - Solución: Volvé a correr el instalador:
    ```bash
    cd "Instalar Requesitos"
    python InstalarTodo.py
    ```

---

## ⏱️ ¿Cuánto tarda en entrenar?

| GPU         | Tiempo estimado | Batch Size |
|-------------|-----------------|------------|
| RTX 3070    | 15-20 min       | 1-2        |
| RTX 4070 Ti | 8-12 min        | 2-4        |
| RTX 4090    | 3-5 min         | 8-16       |

---

## 📤 ¿Dónde queda mi modelo entrenado?

Cuando termina, lo vas a encontrar en:
```
outputs/mistral_finetuned_[config]/
├── adapter_model.safetensors  # Modelo LoRA
├── tokenizer.json             # Tokenizador
└── unsloth.Q4_K_M.gguf       # Formato optimizado
```

---

## 🆘 ¿Algo no anda?

- 🐛 **Problemas**: Corré `python VerificarTodo.py` y fijate qué te dice
- 📖 **Documentación completa**: Mirá el `README.md`
- 🔄 **Reinstalar todo**: Corré `python InstalarTodo.py` de nuevo

---

¿Todo funcionando? ¡Dale una ⭐ al repo y contale a tus amigos! 😎
