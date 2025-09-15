#!/usr/bin/env python3
"""
InstalarTodo.py - Instalador automático para el proyecto Mistral Fine-tuning

Este script instala TODAS las dependencias necesarias para ejecutar
fine-tuning de Mistral-7B con Unsloth de forma automática.

Estructura del proyecto:
- Instalar Requesitos/ (este script)
- RealizarFineTuning/ (script principal)
- Dataset de Miramar/ (datasets personalizados)

Requisitos del sistema:
- GPU NVIDIA con drivers instalados
- CUDA 11.8+ o 12.0+
- Python 3.8+
- Conexión a internet estable
"""

import os
import sys
import subprocess
import platform
import importlib.util

def print_banner():
    """Imprime el banner del instalador"""
    print("=" * 70)
    print("🚀 INSTALADOR AUTOMÁTICO PARA MISTRAL FINE-TUNING")
    print("   Proyecto organizado con estructura de carpetas")
    print("   Instala todo lo necesario para el fine-tuning")
    print("=" * 70)
    print("📁 Estructura del proyecto:")
    print("   ├── Instalar Requesitos/ (instalación)")
    print("   ├── RealizarFineTuning/ (entrenamiento)")
    print("   ├── Dataset de Miramar/ (datos)")
    print("   └── outputs/ (resultados)")
    print("=" * 70)
    print()

def print_section(title):
    """Imprime una sección con formato"""
    print("\n" + "🔥 " + "=" * 60)
    print(f"   {title}")
    print("=" * 65)

def check_python_version():
    """Verifica la versión de Python"""
    print_section("VERIFICANDO VERSIÓN DE PYTHON")
    
    version = sys.version_info
    print(f"🐍 Python detectado: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print("💡 Instala una versión más reciente de Python")
        return False
    
    print("✅ Versión de Python compatible")
    return True

def check_gpu():
    """Verifica la disponibilidad de GPU NVIDIA"""
    print_section("VERIFICANDO GPU NVIDIA")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ No se detectó GPU NVIDIA o drivers no instalados")
            print("💡 Instala los drivers NVIDIA desde: https://www.nvidia.com/drivers")
            print("⚠️  El fine-tuning será MUY lento sin GPU")
            
            choice = input("\n¿Continuar sin GPU? (s/n): ").strip().lower()
            return choice in ['s', 'si', 'sí', 'y', 'yes']
        else:
            # Extraer información de la GPU
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    print(f"✅ GPU detectada: {line.strip()}")
                    break
            
            # Verificar memoria
            for line in lines:
                if 'MiB' in line and '/' in line:
                    print(f"📊 Memoria GPU: {line.strip()}")
                    break
            
            return True
            
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
        print("💡 Instala CUDA toolkit y drivers NVIDIA")
        return False

def check_cuda():
    """Verifica la instalación de CUDA"""
    print_section("VERIFICANDO CUDA")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extraer versión de CUDA
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    print(f"✅ CUDA detectado: {line.strip()}")
                    return True
        else:
            print("⚠️  nvcc no encontrado, pero esto es opcional")
            print("💡 PyTorch puede usar CUDA sin nvcc en algunos casos")
            return True
            
    except FileNotFoundError:
        print("⚠️  CUDA toolkit no detectado, pero puede funcionar")
        print("💡 PyTorch incluye su propia versión de CUDA")
        return True

def run_command(command, description, critical=True):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔧 {description}...")
    print(f"💻 Comando: {command}")
    
    try:
        # Ejecutar comando
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        
        print("✅ Instalación exitosa")
        
        # Mostrar output si es relevante
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 5:
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
            else:
                print(f"   ... {len(lines)} líneas de output")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en: {description}")
        print(f"💀 Código de error: {e.returncode}")
        
        if e.stderr:
            print(f"📄 Error detallado:")
            error_lines = e.stderr.strip().split('\n')
            for line in error_lines[-3:]:  # Mostrar últimas 3 líneas
                print(f"   {line}")
        
        if critical:
            print("🛑 Este error es crítico, deteniendo instalación")
            return False
        else:
            print("⚠️  Error no crítico, continuando...")
            return True

def install_pytorch():
    """Instala PyTorch con soporte CUDA"""
    print_section("INSTALANDO PYTORCH CON CUDA")
    
    # Comando optimizado para CUDA 12.1
    pytorch_cmd = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    
    return run_command(pytorch_cmd, "Instalando PyTorch con CUDA 12.1")

def install_unsloth():
    """Instala Unsloth y dependencias relacionadas"""
    print_section("INSTALANDO UNSLOTH")
    
    # Instalar desde git (versión más actualizada)
    unsloth_cmd = 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    
    if not run_command(unsloth_cmd, "Instalando Unsloth desde GitHub", critical=False):
        print("💡 Intentando instalación desde PyPI...")
        fallback_cmd = "pip install unsloth"
        return run_command(fallback_cmd, "Instalando Unsloth desde PyPI")
    
    return True

def install_transformers_ecosystem():
    """Instala Transformers y librerías relacionadas"""
    print_section("INSTALANDO TRANSFORMERS Y DEPENDENCIAS")
    
    packages = [
        "transformers>=4.40.0",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "scipy",
        "scikit-learn"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Instalando {package}"):
            return False
    
    return True

def install_compatibility_fixes():
    """Instala versiones específicas para compatibilidad"""
    print_section("APLICANDO FIXES DE COMPATIBILIDAD")
    
    # Versiones específicas para evitar conflictos
    fixes = [
        "protobuf==3.20.3",
        "sentencepiece==0.1.99",
        "numpy<2.0.0",
        "packaging",
        "psutil"
    ]
    
    for fix in fixes:
        if not run_command(f"pip install {fix}", f"Instalando {fix}"):
            return False
    
    return True

def verify_installation():
    """Verifica que todas las librerías se importan correctamente"""
    print_section("VERIFICANDO INSTALACIÓN")
    
    # Lista de imports críticos
    critical_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("unsloth", "Unsloth")
    ]
    
    all_passed = True
    
    for module_name, display_name in critical_imports:
        try:
            # Intentar importar
            if module_name == "unsloth":
                # Import especial para Unsloth
                import unsloth
                from unsloth import FastLanguageModel
                print(f"✅ {display_name}: OK")
            else:
                __import__(module_name)
                print(f"✅ {display_name}: OK")
                
        except ImportError as e:
            print(f"❌ {display_name}: FALLO ({str(e)[:50]}...)")
            all_passed = False
        except Exception as e:
            print(f"⚠️  {display_name}: Error inesperado ({str(e)[:30]}...)")
    
    # Verificar CUDA en PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA en PyTorch: OK (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("⚠️  CUDA en PyTorch: No disponible (funcionará con CPU)")
    except:
        print("❌ Error verificando CUDA en PyTorch")
        all_passed = False
    
    return all_passed

def create_test_script():
    """Crea un script de prueba rápida"""
    print_section("CREANDO SCRIPT DE PRUEBA")
    
    test_content = '''#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que todo funciona
"""

def test_imports():
    """Prueba todas las importaciones"""
    print("🧪 Probando importaciones...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        
        import unsloth
        from unsloth import FastLanguageModel
        print("✅ Unsloth importado correctamente")
        
        from transformers import AutoTokenizer
        print("✅ Transformers OK")
        
        from datasets import Dataset
        print("✅ Datasets OK")
        
        from trl import SFTTrainer
        print("✅ TRL OK")
        
        import bitsandbytes
        print("✅ BitsAndBytes OK")
        
        print("\\n🎉 ¡TODAS LAS IMPORTACIONES EXITOSAS!")
        print("🚀 ¡Listo para ejecutar el fine-tuning!")
        print("📁 Ve a la carpeta RealizarFineTuning/ y ejecuta:")
        print("   cd ../RealizarFineTuning")
        print("   python mistral_finetuning_final.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
'''
    
    try:
        with open("test_instalacion.py", "w", encoding="utf-8") as f:
            f.write(test_content)
        print("✅ Script de prueba creado: test_instalacion.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script de prueba: {e}")
        return False

def show_final_instructions():
    """Muestra las instrucciones finales"""
    print_section("INSTALACIÓN COMPLETADA")
    
    print("🎉 ¡INSTALACIÓN EXITOSA!")
    print()
    print("📋 PRÓXIMOS PASOS:")
    print()
    print("1️⃣  Verificar instalación:")
    print("   python VerificarTodo.py")
    print()
    print("2️⃣  Ejecutar fine-tuning:")
    print("   cd ../RealizarFineTuning")
    print("   python mistral_finetuning_final.py")
    print()
    print("📁 ESTRUCTURA DEL PROYECTO:")
    print("   ├── Instalar Requesitos/")
    print("   │   ├── InstalarTodo.py (este archivo)")
    print("   │   └── VerificarTodo.py (verificador)")
    print("   ├── RealizarFineTuning/")
    print("   │   └── mistral_finetuning_final.py (script principal)")
    print("   ├── Dataset de Miramar/")
    print("   │   └── *.jsonl (tu dataset personalizado)")
    print("   └── outputs/ (modelos entrenados)")
    print()
    print("💡 CONSEJOS:")
    print("   • El primer fine-tuning descargará ~14GB (modelo)")
    print("   • Asegúrate de tener 50GB de espacio libre")
    print("   • El proceso completo toma 5-8 minutos en GPU")
    print()
    print("🆘 SI HAY PROBLEMAS:")
    print("   • Ejecuta: python3 test_instalacion.py")
    print("   • Verifica que la GPU esté disponible")
    print("   • Reinicia el terminal si es necesario")
    print()
    print("🚀 ¡LISTO PARA ENTRENAR MODELOS!")

def main():
    """Función principal del instalador"""
    print_banner()
    
    # 1. Verificaciones del sistema
    if not check_python_version():
        sys.exit(1)
    
    gpu_available = check_gpu()
    check_cuda()
    
    # Confirmar instalación
    print("\n" + "⚠️ " * 20)
    print("ESTE INSTALADOR VA A:")
    print("• Instalar PyTorch (~2GB)")
    print("• Instalar Unsloth y dependencias (~1GB)")
    print("• Instalar Transformers ecosystem (~500MB)")
    print("• Aplicar fixes de compatibilidad")
    print("• Total estimado: ~3.5GB de descarga")
    print("⚠️ " * 20)
    
    if not gpu_available:
        print("\n🚨 ADVERTENCIA: Sin GPU el entrenamiento será EXTREMADAMENTE lento")
    
    continue_install = input("\n¿Continuar con la instalación? (s/n): ").strip().lower()
    
    if continue_install not in ['s', 'si', 'sí', 'y', 'yes']:
        print("❌ Instalación cancelada")
        sys.exit(0)
    
    print("\n🚀 Iniciando instalación automática...")
    
    # 2. Instalar dependencias
    steps = [
        (install_pytorch, "PyTorch con CUDA"),
        (install_transformers_ecosystem, "Transformers y dependencias"),
        (install_unsloth, "Unsloth"),
        (install_compatibility_fixes, "Fixes de compatibilidad"),
    ]
    
    for step_func, step_name in steps:
        print(f"\n📦 Instalando {step_name}...")
        if not step_func():
            print(f"\n💥 FALLO EN: {step_name}")
            print("🛑 Instalación detenida")
            sys.exit(1)
    
    # 3. Verificar instalación
    print(f"\n🔍 Verificando instalación completa...")
    if not verify_installation():
        print("\n⚠️  Algunas verificaciones fallaron")
        print("💡 Pero puede que aún funcione. Prueba ejecutar:")
        print("   python3 test_instalacion.py")
    
    # 4. Crear script de prueba
    create_test_script()
    
    # 5. Mostrar instrucciones finales
    show_final_instructions()

if __name__ == "__main__":
    main()
