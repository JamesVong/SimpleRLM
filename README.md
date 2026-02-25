# Overview
Basic implementation of RLMs from *Recursive Language Models, Zhang et al (2026).* [Link here.](https://arxiv.org/pdf/2512.24601)

# Installation

Requires a .env file that contains an NVIDIA API key, which you can get [here](https://build.nvidia.com/models).
```env
NVIDIA_API_KEY=nvapi-YOUR_KEY
```

To run the code, you need to install the required libraries first:
```
pip install -r requirements.txt
streamlit run app.py
```