# LAB04-DL-2026-01

Reconocimiento multitarea de género y edad con aumento de datos tradicional, autoencoders, VAE y GAN usando UTKFace.

## Objetivo

Este laboratorio extiende la lógica del LAB03 para comparar un modelo CNN multitarea entrenado con imágenes reales y distintas fuentes de aumento de datos. La tarea de género se aborda como clasificación y la tarea de edad como regresión por defecto.

## Datos originales

Las imágenes originales de UTKFace se cargan desde la ruta configurable:

```yaml
paths:
  original_utkface_dir: data/raw/UTKFace
```

Esa carpeta es la única fuente de imágenes originales y debe tratarse como entrada de solo lectura. Si el dataset está en otra ubicación, cambie solo `paths.original_utkface_dir` en `config/path.yaml`.

## Imágenes generadas

Las imágenes sintéticas se guardan localmente dentro del proyecto:

```text
data/generated/cae/
data/generated/vae/
data/generated/gan/
```

Sus metadatos se guardan en `data/metadata/` con la columna `source`, para distinguir imágenes `real`, `cae`, `vae` y `gan`.

Los scripts de generación reutilizan imágenes existentes por defecto si la carpeta y su CSV de metadatos son válidos. Para exigir que ya existan imágenes generadas:

```bash
python3 scripts/03_generate_cae_faces.py --config config/path.yaml --use-existing
python3 scripts/05_generate_vae_faces.py --config config/path.yaml --use-existing
python3 scripts/07_generate_gan_faces.py --config config/path.yaml --use-existing
```

Para regenerar aunque ya existan, use `--force-generate`. La generación usa semillas configurables en `generation.seed`, `generation.cae_seed`, `generation.vae_seed` y `generation.gan_seed`; también se puede sobrescribir por comando:

```bash
python3 scripts/05_generate_vae_faces.py --config config/path.yaml --force-generate --seed 2026
```

## Instalación

Con Conda:

```bash
conda env create -f environment.yml
conda activate lab04-dl-2026-01
```

Con `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Orquestador principal

El archivo `main.py` es el punto de entrada principal del laboratorio. Internamente llama a los scripts numerados para mantener el flujo transparente.

```bash
python3 main.py list
python3 main.py prepare
python3 main.py train --experiment E00_real_only
python3 main.py evaluate --experiment E00_real_only
```

Para revisar el flujo sin ejecutarlo:

```bash
python3 main.py --dry-run minimal
python3 main.py --dry-run extended
```

## Flujo mínimo

```bash
python3 main.py minimal
```

El flujo mínimo ejecuta, en orden, los comandos equivalentes a:

```bash
python3 scripts/01_prepare_dataset.py --config config/path.yaml
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E00_real_only
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E01_real_traditional
python3 scripts/02_train_cae.py --config config/path.yaml
python3 scripts/03_generate_cae_faces.py --config config/path.yaml
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E02_real_cae
python3 scripts/09_evaluate_experiments.py --config config/path.yaml
```

Para usar imágenes CAE existentes y fallar si no están disponibles:

```bash
python3 main.py minimal --use-existing-generated
```

## Flujo extendido

```bash
python3 main.py extended
```

El flujo extendido ejecuta, en orden, los comandos equivalentes a:

```bash
python3 scripts/04_train_vae.py --config config/path.yaml
python3 scripts/05_generate_vae_faces.py --config config/path.yaml
python3 scripts/06_train_gan.py --config config/path.yaml
python3 scripts/07_generate_gan_faces.py --config config/path.yaml
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E03_real_vae
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E04_real_gan
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E05_all
python3 scripts/09_evaluate_experiments.py --config config/path.yaml
```

Para reutilizar imágenes VAE/GAN existentes sin entrenar sus generadores:

```bash
python3 main.py extended --use-existing-generated
```

Para regenerar imágenes sintéticas con una semilla fija:

```bash
python3 main.py extended --force-generate --seed 2026
```

## Experimentos

Los experimentos están definidos en `config/experiments.yaml`:

- `E00_real_only`: imágenes reales.
- `E01_real_traditional`: imágenes reales con aumento tradicional.
- `E02_real_cae`: imágenes reales más imágenes CAE.
- `E03_real_vae`: imágenes reales más imágenes VAE.
- `E04_real_gan`: imágenes reales más imágenes GAN.
- `E05_all`: combinación completa.

Validación y prueba usan solo imágenes reales. Las imágenes generadas se mezclan únicamente con el conjunto de entrenamiento.

## Búsqueda simple de hiperparámetros

El laboratorio incluye un barrido pequeño y explícito en `config/experiments.yaml`, bajo la clave `hyperparameter_sweeps`.

Para ver los barridos disponibles:

```bash
python3 main.py sweep --list
```

Para revisar los comandos sin ejecutarlos:

```bash
python3 main.py --dry-run sweep --name quick_multitask
```

Para ejecutar el barrido:

```bash
python3 main.py sweep --name quick_multitask
```

El barrido `quick_multitask` prueba pocas combinaciones didácticas sobre el experimento base `E00_real_only`, por ejemplo:

- `losses.lambda_age: 0.5`
- `losses.lambda_age: 1.0`
- `losses.lambda_age: 2.0`
- `training.learning_rate: 0.0005` con `multitask_model.dropout: 0.2`
- `training.learning_rate: 0.001` con `multitask_model.dropout: 0.5`

Cada corrida se guarda como un experimento independiente, por ejemplo `hp_lambda_age_05`, `hp_lambda_age_10` o `hp_lr_001_dropout_05`.

## Métricas

El laboratorio reporta métricas separadas por tarea:

- Género: accuracy, precision, recall y F1.
- Edad: MAE y RMSE.
- Global: pérdida total, pérdida de género y pérdida de edad.

## Material docente

El archivo `presentation/Laboratorio_04_Guia_Uso_Codigo.tex` contiene una presentación Beamer didáctica para guiar el uso del laboratorio en servidor, explicar los YAML y revisar los comandos principales.
