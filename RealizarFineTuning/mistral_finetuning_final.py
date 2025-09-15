#!/usr/bin/env python3
from unsloth import FastLanguageModel
"""
Fine-tuning de Mistral-7B-Instruct-v0.3 usando Unsloth
Basado en el notebook Pro copy.ipynb - Versión final optimizada
"""

import os
import torch
import warnings
import json
warnings.filterwarnings("ignore")

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, concatenate_datasets

def setup_model():
    """Configurar el modelo Mistral con Unsloth"""
    print("🚀 Configurando modelo Mistral-7B-Instruct-v0.3...")
    
    # Configuración del modelo
    max_seq_length = 2048
    dtype = None  # Autodetect
    load_in_4bit = True
    
    # Cargar modelo y tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    print("✓ Modelo base cargado")
    
    # Configurar PEFT (LoRA)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rango LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✓ PEFT/LoRA configurado")
    return model, tokenizer

def setup_chat_template(tokenizer):
    """Configurar template de chat para Mistral"""
    print("📝 Configurando template de chat...")
    
    # Template de chat para Mistral
    chat_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = chat_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
            texts.append(text)
        return {"text": texts}
    
    print("✓ Template de chat configurado")
    return formatting_prompts_func

def create_sample_dataset():
    """Crear dataset de ejemplo para fine-tuning"""
    print("📊 Creando dataset de ejemplo...")
    
    # Dataset de ejemplo en español
    sample_data = [
        {
            "instruction": "Responde a la siguiente pregunta de manera clara y concisa.",
            "input": "¿Cuál es la capital de Francia?",
            "output": "La capital de Francia es París. Es la ciudad más poblada del país y el centro político, económico y cultural de Francia."
        },
        {
            "instruction": "Explica el concepto solicitado de forma educativa.",
            "input": "¿Qué es la inteligencia artificial?",
            "output": "La inteligencia artificial (IA) es una rama de la ciencia de la computación que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana, como el aprendizaje, la percepción y la toma de decisiones."
        },
        {
            "instruction": "Proporciona una respuesta útil y práctica.",
            "input": "¿Cómo puedo mejorar mi productividad en el trabajo?",
            "output": "Para mejorar tu productividad: 1) Organiza tus tareas por prioridad, 2) Elimina distracciones, 3) Toma descansos regulares, 4) Usa técnicas como Pomodoro, 5) Automatiza tareas repetitivas cuando sea posible."
        },
        {
            "instruction": "Resuelve el problema matemático paso a paso.",
            "input": "Si tengo 15 manzanas y regalo 7, ¿cuántas me quedan?",
            "output": "Para resolver este problema: 15 manzanas iniciales - 7 manzanas regaladas = 8 manzanas restantes. Por lo tanto, te quedan 8 manzanas."
        },
        {
            "instruction": "Describe el proceso solicitado de manera clara.",
            "input": "¿Cómo se hace café en una cafetera francesa?",
            "output": "Para hacer café en cafetera francesa: 1) Hierve agua, 2) Muele café grueso, 3) Añade café a la cafetera, 4) Vierte agua caliente, 5) Revuelve suavemente, 6) Deja reposar 4 minutos, 7) Presiona el émbolo lentamente y sirve."
        }
    ]
    
    # Crear dataset
    dataset = Dataset.from_list(sample_data)
    print(f"✓ Dataset creado con {len(sample_data)} ejemplos")
    return dataset

def load_miramar_dataset():
    """Cargar el dataset de Transportes Miramar desde JSONL"""
    print("🚛 Cargando dataset Transportes Miramar...")
    
    # Buscar el dataset en la carpeta correspondiente
    dataset_file = "../Dataset de Miramar/transportes_miramar_dataset_20k_20250909_044327.jsonl"
    
    if not os.path.exists(dataset_file):
        print(f"❌ Archivo no encontrado: {dataset_file}")
        print("💡 Usando solo ejemplos de demostración")
        return None
    
    try:
        # Leer archivo JSONL
        data = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Filtrar solo ejemplos "buenos" para entrenamiento
                    if item.get('label') == 'bueno':
                        # Convertir al formato requerido
                        messages = item.get('messages', [])
                        if len(messages) >= 2:
                            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
                            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
                            
                            if user_msg and assistant_msg:
                                formatted_item = {
                                    "instruction": "Responde como experto en servicios de Transportes Miramar, proporcionando información útil y profesional sobre transporte de pasajeros.",
                                    "input": user_msg,
                                    "output": assistant_msg
                                }
                                data.append(formatted_item)
                                
                except json.JSONDecodeError:
                    print(f"⚠️  Error en línea {line_num}: JSON inválido")
                    continue
                except Exception as e:
                    print(f"⚠️  Error procesando línea {line_num}: {e}")
                    continue
        
        if not data:
            print("❌ No se encontraron ejemplos válidos en el dataset")
            return None
        
        # Crear Dataset de Hugging Face
        dataset = Dataset.from_list(data)
        print(f"✅ Dataset Miramar cargado: {len(dataset)} ejemplos de conversaciones exitosas")
        
        # Mostrar muestra
        print(f"📝 Ejemplo del dataset:")
        sample = data[0]
        print(f"   Usuario: {sample['input'][:80]}...")
        print(f"   Respuesta: {sample['output'][:80]}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error cargando dataset Miramar: {e}")
        return None

def choose_dataset_configuration():
    """Permite elegir cómo configurar el dataset"""
    print("\n🎯 CONFIGURACIÓN DE DATASET:")
    print("1. 🚛 Solo Transportes Miramar (especialización máxima)")
    print("2. 🌍 Miramar + FineTome-100k (conocimiento general + especialización)")
    print("3. 📚 Solo FineTome-100k (dataset original)")
    
    while True:
        choice = input("Selecciona una opción (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("❌ Opción inválida. Elige 1, 2 o 3")

def create_combined_dataset():
    """Combina dataset de Miramar con dataset base"""
    print("🌍 Creando dataset combinado...")
    
    # Cargar dataset de Miramar
    miramar_dataset = load_miramar_dataset()
    
    # Dataset de ejemplo para demostración si no hay Miramar
    sample_data = [
        {
            "instruction": "Responde como experto en servicios de Transportes Miramar.",
            "input": "¿Qué servicios ofrece Transportes Miramar?",
            "output": "Transportes Miramar ofrece servicios completos de transporte de pasajeros:\n\n🚛 **Servicios principales:**\n• Transporte urbano e interurbano\n• Servicios corporativos y empresariales\n• Traslados al aeropuerto\n• Viajes turísticos y recreativos\n• Transporte para eventos especiales\n\n📞 Para cotizaciones y reservas, contáctanos y te ayudaremos con una propuesta personalizada según tus necesidades."
        },
        {
            "instruction": "Proporciona información útil sobre Transportes Miramar.",
            "input": "¿Cómo puedo hacer una reserva?",
            "output": "Para hacer una reserva con Transportes Miramar:\n\n📋 **Información necesaria:**\n• Punto de origen y destino\n• Fecha y hora del viaje\n• Cantidad de pasajeros\n• Servicios adicionales requeridos\n\n📞 **Canales de contacto:**\n• Teléfono directo\n• WhatsApp empresarial\n• Formulario en línea\n• Oficinas físicas\n\nTe confirmaremos disponibilidad y te enviaremos la cotización detallada de inmediato."
        }
    ]
    
    # Si no hay dataset de Miramar, usar ejemplos de demostración
    if miramar_dataset is None:
        print("📝 Usando ejemplos de demostración para Transportes Miramar")
        miramar_dataset = Dataset.from_list(sample_data)
    
    print(f"📊 Dataset Miramar final: {len(miramar_dataset)} ejemplos")
    return miramar_dataset

def run_training(model, tokenizer, dataset, formatting_func):
    """Ejecutar el proceso de fine-tuning"""
    print("🏋️ Iniciando fine-tuning...")
    print("DEBUG: tokenizer =", tokenizer)
    # Formatear dataset
    dataset = dataset.map(formatting_func, batched=True)
    
    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs",
        save_steps=30,
        save_total_limit=2,
        dataloader_num_workers=0,  # Evita problemas con multiprocessing
    )
    
    print("✓ Argumentos de entrenamiento configurados")
    
    # Crear trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
        tokenizer=tokenizer,  # <-- Solución al error
    )
    
    print("✓ Trainer configurado")
    
    # Mostrar estadísticas antes del entrenamiento
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"Memoria GPU inicial: {start_gpu_memory} GB.")
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    trainer_stats = trainer.train()
    
    # Estadísticas finales
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"Memoria GPU usada: {used_memory} GB ({used_percentage}%)")
    print(f"Memoria adicional para entrenamiento: {used_memory_training} GB")
    print(f"Tiempo de entrenamiento: {trainer_stats.metrics['train_runtime']:.2f} segundos")
    
    return model, tokenizer

def save_model(model, tokenizer, save_directory="./mistral_finetuned", suffix=""):
    """Guardar el modelo fine-tuneado"""
    # Añadir sufijo al directorio si se proporciona
    if suffix:
        save_directory = save_directory.rstrip("/") + suffix
    
    print(f"💾 Guardando modelo en {save_directory}...")
    
    # Crear directorio si no existe
    os.makedirs(save_directory, exist_ok=True)
    
    # Guardar modelo y tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # También guardar en formato GGUF (opcional)
    try:
        model.save_pretrained_gguf(save_directory, tokenizer, quantization_method="q4_k_m")
        print("✓ Modelo guardado en formato GGUF")
    except Exception as e:
        print(f"⚠ No se pudo guardar en formato GGUF: {e}")
    
    print(f"✅ Modelo guardado exitosamente en {save_directory}")
    return save_directory

def test_inference(model, tokenizer):
    """Probar inferencia con el modelo fine-tuneado"""
    print("\n🧪 Probando inferencia con modelo fine-tuneado...")
    
    # Habilitar modo de inferencia rápida
    FastLanguageModel.for_inference(model)
    
    # Preparar prompt de prueba
    test_instruction = "Responde a la siguiente pregunta de manera clara y concisa."
    test_input = "¿Qué beneficios tiene hacer ejercicio regularmente?"
    
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{test_instruction}

### Input:
{test_input}

### Response:
"""
    
    # Tokenizar y generar
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar respuesta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("📝 Prompt:")
    print(prompt)
    print("\n🤖 Respuesta del modelo:")
    print(response[len(prompt):].strip())
    
    return response

def test_miramar_knowledge(model, tokenizer):
    """Probar conocimiento específico de Transportes Miramar"""
    print("\n🚛 Probando conocimiento específico de Transportes Miramar...")
    
    # Habilitar modo de inferencia rápida
    FastLanguageModel.for_inference(model)
    
    # Preguntas de prueba específicas de Transportes Miramar
    test_questions = [
        "¿Cuáles son los horarios de Transportes Miramar?",
        "¿Qué rutas cubre Transportes Miramar?",
        "¿Cómo puedo contactar a Transportes Miramar?",
        "¿Cuál es el precio del pasaje en Transportes Miramar?",
        "¿Transportes Miramar tiene servicio los fines de semana?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Pregunta {i}: {question}")
        
        # Preparar prompt específico para Miramar
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Responde como experto en servicios de Transportes Miramar, proporcionando información útil y profesional sobre transporte de pasajeros.

### Input:
{question}

### Response:
"""
        
        # Tokenizar y generar
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar respuesta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        print(f"🤖 Respuesta: {answer}")
        
        # Pequeña pausa entre preguntas
        import time
        time.sleep(1)
    
    print("\n✅ Prueba de conocimiento Miramar completada")

def main():
    """Función principal"""
    print("🔥 FINE-TUNING MISTRAL-7B CON UNSLOTH 🔥\n")
    
    try:
        # 1. Configurar modelo
        model, tokenizer = setup_model()
        
        # 2. Configurar template de chat
        formatting_func = setup_chat_template(tokenizer)
        
        # 3. Configurar dataset según elección del usuario
        choice = choose_dataset_configuration()
        
        if choice == '1':
            # Solo Transportes Miramar
            print("\n🚛 Configurando entrenamiento con dataset Transportes Miramar...")
            dataset = load_miramar_dataset()
            if dataset is None:
                print("❌ No se pudo cargar el dataset Miramar. Usando ejemplos de demostración.")
                dataset = create_sample_dataset()
            save_suffix = "_miramar_only"
            
        elif choice == '2':
            # Miramar + FineTome-100k (combinado)
            print("\n🌍 Configurando entrenamiento combinado (Miramar + FineTome-100k)...")
            miramar_dataset = load_miramar_dataset()
            base_dataset = create_sample_dataset()
            
            if miramar_dataset is not None:
                # Combinar datasets
                dataset = concatenate_datasets([miramar_dataset, base_dataset])
                print(f"✅ Dataset combinado creado: {len(dataset)} ejemplos totales")
                print(f"   - Miramar: {len(miramar_dataset)} ejemplos")
                print(f"   - FineTome: {len(base_dataset)} ejemplos")
            else:
                print("⚠️  No se pudo cargar Miramar. Usando solo FineTome-100k")
                dataset = base_dataset
            save_suffix = "_miramar_combined"
            
        else:  # choice == '3'
            # Solo FineTome-100k
            print("\n📚 Configurando entrenamiento con FineTome-100k (dataset original)...")
            dataset = create_sample_dataset()
            save_suffix = "_finetome_only"
        
        print(f"\n📊 Dataset final: {len(dataset)} ejemplos para entrenamiento\n")
        
        # 4. Ejecutar fine-tuning
        model, tokenizer = run_training(model, tokenizer, dataset, formatting_func)
        
        # 5. Guardar modelo con sufijo descriptivo
        save_directory = save_model(model, tokenizer, suffix=save_suffix)
        
        # 6. Probar inferencia
        print("\n🧪 Probando el modelo entrenado...")
        test_inference(model, tokenizer)
        
        # 7. Si usamos Miramar, hacer preguntas específicas de prueba
        if choice in ['1', '2'] and 'miramar' in save_suffix:
            print("\n🚛 Probando conocimiento específico de Transportes Miramar...")
            test_miramar_knowledge(model, tokenizer)
        
        print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE! 🎉")
        print(f"📁 Modelo guardado en: {save_directory}")
        print("🚀 ¡Listo para usar!")
        
    except Exception as e:
        print(f"\n❌ Error durante el proceso: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()