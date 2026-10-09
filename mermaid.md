# 🗺️ Synthèse Architecture PatchTST

```mermaid
graph TD
    A[🧠 Du Transformer NLP à PatchTST] --> B[📜 1. Héritage NLP & Limites]
    A --> C[🏗️ 2. Piliers de PatchTST]
    A --> D[📐 3. Parcours Matriciel]
    A --> E[⚙️ 4. Apprentissage & Compromis]

    B --> B1[📍 Traitement point par point]
    B --> B2[⏳ Complexité O-L²]

    C --> C1[🧩 Patching P]
    C --> C2[🔀 Indépendance des Canaux CI]
    C --> C3[🤝 Weight Sharing]

    D --> D1[📥 Entrée: 1, 3, 100]
    D --> D2[🧩 Patching: 3, 10, 10]
    D --> D3[⚡ Embedding: 3, 10, 32]
    D --> D4[📤 Sortie: 1, 3, 24]

    E --> E1[🎯 Loss MSE]
    E --> E2[🔄 Rétropropagation]
    E --> E3[⚠️ Pas de corrélations inter-canaux]