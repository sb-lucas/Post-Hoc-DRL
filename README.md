# Interpretando Similaridade Musical em Modelos Fundacionais via Decomposição de Subespaços
Decomposição de representação de músicas em diferentes sub representações específicas.

Trabalho feito por:

- João Brasileiro Moreira da Silva - 13672957
- João Gabriel Manfré Nazar - 13733652
- Lucas de Souza Brandão - 13695021
- Vicenzo D'Arezzo Zilio - 13671790

## Instalação

As dependências podem ser instaladas utilizando Conda ou pip.

### Usando Conda
```bash
conda env create -f environment.yml
conda activate csn-music
```

### Usando pip
```bash
pip install -r requirements.txt
```

## Estrutura

- `dataset/` - contém scripts para extração de embeddings e classificação das tags das músicas, bem como código para criar triplets
- `csn/` - implementação do CSN e máscaras, bem como código para treinar. Contém a pasta `training_helpers` com arquivos úteis para treinamento.
- `evaluation/` - scripts para avaliar cada uma das métricas