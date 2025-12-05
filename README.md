# 🔬 Mistral 3 Lab

Et interaktivt Streamlit-interface til at eksperimentere med Mistral AI's nye Ministral 3 modeller (3B, 8B, 14B) lokalt via Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🤖 Model Valg** - Skift nemt mellem Ministral 3B, 8B og 14B modeller
- **⚙️ Justerbare Settings** - Temperature, Top P, Max Tokens, Repeat Penalty
- **💬 Chat Interface** - Streaming responses med real-time token statistik
- **🔄 Model Sammenligning** - Sammenlign alle modeller side-by-side med samme prompt
- **📝 Preset Prompts** - Indbyggede og custom prompt templates
- **📊 Token Tæller** - Hold styr på forbrug i din session

## 📋 Forudsætninger

### Hardware
- **GPU:** NVIDIA GPU med mindst 16GB VRAM (24GB+ anbefales for alle modeller)
- **RAM:** Minimum 8GB system RAM

### Software
- **Python:** 3.10 eller nyere
- **Ollama:** Installeret og kørende

## 🚀 Installation

### 1. Installer Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download Ministral 3 modeller

```bash
# Vælg én eller flere modeller efter dit VRAM budget:
ollama pull ministral-3:3b   # ~13GB VRAM - Hurtigst
ollama pull ministral-3:8b   # ~16GB VRAM - Balanceret
ollama pull ministral-3:14b  # ~14GB VRAM - Bedst kvalitet
```

### 3. Klon dette repository

```bash
git clone https://github.com/Broadhead-Logic/ministral3-demo.git
cd ministral3-demo
```

### 4. Opret virtual environment og installer dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 💻 Brug

### Start applikationen

```bash
# Sørg for at Ollama kører
ollama serve

# I en ny terminal, start Streamlit
source venv/bin/activate
streamlit run app.py
```

Åbner automatisk i din browser på: **http://localhost:8501**

### Interface Guide

#### Sidebar (Venstre)
- **Model:** Vælg mellem tilgængelige Ministral modeller
- **Generation Settings:** Juster temperature, top_p, max tokens, repeat penalty
- **System Prompt:** Tilpas AI'ens persona
- **Preset Prompts:** Vælg fra foruddefinerede prompt templates
- **Session Stats:** Se dit token forbrug

#### Chat Tab
- Skriv beskeder og få streaming responses
- Se token count og hastighed for hver besked

#### Sammenligning Tab
- Send samme prompt til alle installerede modeller
- Sammenlign svar, hastighed og kvalitet side-by-side

## ⚙️ Konfiguration

### Ollama Settings

Du kan justere hvor længe modeller forbliver i VRAM:

```bash
# Unload efter 1 minut inaktivitet (default: 5m)
OLLAMA_KEEP_ALIVE=1m ollama serve

# Unload straks efter hver request
OLLAMA_KEEP_ALIVE=0 ollama serve

# Hold altid i VRAM
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

### Model Performance

| Model | VRAM | Hastighed* | Kvalitet |
|-------|------|-----------|----------|
| Ministral 3B | ~13GB | ~360 tok/s | God |
| Ministral 8B | ~16GB | ~240 tok/s | Meget god |
| Ministral 14B | ~14GB | ~40 tok/s | Bedst |

*Hastigheder målt på RTX 5090. Din performance vil variere.

**Vigtigt:** Ollama holder modeller i VRAM i 5 minutter efter brug. Ved hurtig skift mellem modeller kan VRAM bruges af flere modeller samtidigt. Brug `OLLAMA_KEEP_ALIVE=0 ollama serve` for at undgå dette.

## 🛠️ Teknisk Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** [Ollama](https://ollama.com/) REST API
- **Modeller:** [Mistral AI Ministral 3](https://mistral.ai/news/mistral-3)

## 📁 Projektstruktur

```
mistral3-lab/
├── app.py              # Hoved Streamlit applikation
├── requirements.txt    # Python dependencies
├── README.md          # Denne fil
└── LICENSE            # MIT License
```

## 🤝 Bidrag

Bidrag er velkomne! Åbn gerne en issue eller pull request.

## 📄 Licens

Dette projekt er licenseret under MIT License - se [LICENSE](LICENSE) filen for detaljer.

## 🙏 Credits

- [Mistral AI](https://mistral.ai/) for de fantastiske Ministral 3 modeller
- [Ollama](https://ollama.com/) for simpel lokal LLM hosting
- [Streamlit](https://streamlit.io/) for det intuitive web framework
