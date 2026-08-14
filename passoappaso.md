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
## modelos:
More disentanglement strategies besides CSN — yes, and this is probably your highest-value addition. Beyond the CSN family already in your related work ([6],[7],[8]), the literature has moved toward closed-form / training-light subspace methods that are a natural fit for "can we disentangle a frozen model without retraining":
LEACE (Belrose et al., 2023) — closed-form linear concept erasure, provably minimal-distortion, no iterative training. Directly testable as an alternative to your learned ReLU mask.
INLP (Ravfogel et al., 2020) and RLACE (Ravfogel et al., 2022) — iterative/adversarial linear subspace removal, the predecessors LEACE improves on. Good baselines to show CSN isn't the only or best option.
Simple linear probing + orthogonal projection (fit a linear classifier per attribute, project onto its weight direction/null space) as a cheap, interpretable baseline — this also doubles as your "how linearly decodable is each attribute" analysis (see below).
Sparse autoencoders (SAEs) on frozen embeddings — increasingly used in interpretability work to find monosemantic/disentangled directions without touching the base model; worth at least discussing even if you don't have room to run it.

## metricas:
How the literature evaluates disentanglement — you're currently using only cosine similarity between masks + sparsity + downstream triplet accuracy. The disentanglement literature has a more standardized toolkit you should adopt or explicitly justify not using:

MIG (Mutual Information Gap), DCI (Disentanglement / Completeness / Informativeness — Eastwood & Williams 2018), SAP (Separated Attribute Predictability). These require ground-truth factors per sample (which you have: instrument/genre/emotion tags), so they're directly applicable to your masks, unlike VAE-style latent traversal metrics.
Important caveat from the literature: no single metric is trustworthy alone — DCI, MIG, and SAP frequently disagree and are sensitive to hyperparameters/classifier choice, so if you adopt these, report at least two and discuss disagreement rather than picking whichever looks best.
This also gives you a principled way to answer RQ3 (which you currently can't answer) — DCI's informativeness score per attribute is essentially the tool for "which subspace best explains X."

## Datasets:
mtg jamendo pra treino e magna pra teste mesmo.