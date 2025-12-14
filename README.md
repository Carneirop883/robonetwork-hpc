RoboNetwork HPC

Pipeline profissional de treino, avaliação e exportação de modelos de Machine Learning em ambiente HPC, desenhado para clusters com SLURM, containers Apptainer/Singularity e aceleração GPU (CUDA).

Este repositório contém apenas código, definições e jobs.
Artefactos pesados (containers .sif, datasets, outputs) são geridos fora do Git, conforme boas práticas HPC.

Visão geral

O objetivo deste projeto é fornecer uma base limpa, reprodutível e escalável para:

Treino de modelos PyTorch em HPC

Execução de jobs SLURM de forma controlada

Separação clara entre:

código

definição de containers

execução

outputs

O pipeline foi pensado para funcionar em ambientes como:

Deucalion / EuroHPC

clusters académicos ou industriais

infraestruturas on-prem com SLURM

Estrutura do repositório
robonetwork-hpc/
├── containers/
│   └── base.def                # Definição do container (Apptainer/Singularity)
│
├── jobs/
│   ├── build_base_container.slurm
│   ├── create_torch_env.slurm
│   ├── train.slurm
│   ├── evaluate.slurm
│   ├── export.slurm
│   └── test_gpu.slurm
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── export_model.py
│
├── test_pytorch_container.slurm
├── .gitignore
└── README.md

Princípios do projeto

Git só para texto e definições

.py, .slurm, .def

Nunca versionar binários

.sif, datasets, checkpoints, outputs

Reprodutibilidade

containers gerados a partir de .def

Separação clara de responsabilidades

scripts ≠ jobs ≠ containers

Containers
Definição

Os containers são definidos em:

containers/base.def


Este ficheiro descreve:

base CUDA

dependências do sistema

Python / PyTorch

bibliotecas necessárias ao treino

Build (fora do Git)

O build do container é feito via SLURM:

sbatch jobs/build_base_container.slurm


O resultado será um ficheiro .sif, por exemplo:

cuda_base.sif


📌 Nota importante
Os ficheiros .sif:

não entram no Git

vivem no filesystem do cluster (ex: /projects/...)

ou em storage externo (S3, MinIO, etc.)

Jobs SLURM

Todos os jobs estão na pasta jobs/.

Teste de GPU
sbatch jobs/test_gpu.slurm


Verifica:

acesso a GPU

CUDA disponível

PyTorch funcional

Treino
sbatch jobs/train.slurm


Executa:

scripts/train.py

usando o container definido

com recursos controlados por SLURM

Avaliação
sbatch jobs/evaluate.slurm


Executa:

scripts/evaluate.py

sobre um modelo treinado

Exportação
sbatch jobs/export.slurm


Executa:

scripts/export_model.py

exporta o modelo para formato final (ex: .pt, .onnx)

Scripts Python

Os scripts vivem em scripts/ e devem ser:

independentes do SLURM

independentes do cluster

fáceis de testar localmente (quando possível)

train.py

Responsável por:

carregar dados

treinar o modelo

guardar checkpoints

evaluate.py

Responsável por:

avaliar métricas

gerar resultados de validação

export_model.py

Responsável por:

converter/exportar modelos treinados

preparar artefactos finais

Estrutura recomendada (fora do Git)

Estas pastas não devem ser versionadas, mas são recomendadas no cluster:

datasets/
models/
outputs/
logs/


Normalmente localizadas em:

/projects/<project_id>/robonetwork/

Boas práticas adotadas

Containers reprodutíveis

Jobs isolados e explícitos

Histórico Git limpo

Escalável para múltiplos modelos e experiências

Preparado para automação futura
