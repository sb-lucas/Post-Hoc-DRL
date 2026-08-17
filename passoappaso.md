# divisoa de tarefa
lucas - preprocessar as fita, baixar dataset,, montar dataset
merlin - codar modelo e rodar

### dataset

pega 2 dataset de autotagging (mtg jamendo e magna tag a tune)
agrupa as tags em genero, instrumento e emoção
extrai feature dos audios com foundation models (clamp3, clap, muq) (lembra de tirar a média dos chunks)
cria triplets dos dois datasets


### modelos 
treina/experimenta modelos (roda 5x pelo menos):
- backbone sozinho
- backbone com mascara aleatoria
- backbone com a interseção das mascaras (verdade universal)
- backbone com mascaras do CSN tradicional com e sem finetuning
- backbone com mascaras do CSN + MLP com e sem finetuning


# to do
- procurar mais estrategias alem de CSN para disentanglar
- ver se ja fizeram plug and play
- como aavliar o quão disentagled q ta
- ver se  ta realmente so treinando com o tipo q deveria
- (se for manter mascaras, ve se ohá um subesaço como a todas e se é ele q responde tudo.)


# qq vai ter no novo paper além doq ja tem
- outros modelos/estratégias
- mais backbones, e melhores (muq-mulan (claude falou q o encoder e o output do muq e do muq-mulan são levemente diferentes e ele prefere o mulan), clap e mert)
- cross evaluation
- métrica decente de avaliação + visualização tsne (ou outra, mas parece bom ter avaliação visual)
- comparar com literatura 
- multiplas runs, com +- std
- pro CSN (e talvez para outros), tem que ter TRIPLET MINING


# claude falou:

## metricas:
How the literature evaluates disentanglement — you're currently using only cosine similarity between masks + sparsity + downstream triplet accuracy. The disentanglement literature has a more standardized toolkit you should adopt or explicitly justify not using:

MIG (Mutual Information Gap), DCI (Disentanglement / Completeness / Informativeness — Eastwood & Williams 2018), SAP (Separated Attribute Predictability). These require ground-truth factors per sample (which you have: instrument/genre/emotion tags), so they're directly applicable to your masks, unlike VAE-style latent traversal metrics.
Important caveat from the literature: no single metric is trustworthy alone — DCI, MIG, and SAP frequently disagree and are sensitive to hyperparameters/classifier choice, so if you adopt these, report at least two and discuss disagreement rather than picking whichever looks best.


## Datasets:
mtg jamendo pra treino e magna pra teste mesmo.