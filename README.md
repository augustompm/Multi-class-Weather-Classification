# Classificação de Condições Climáticas com CNN

Este projeto implementa uma **Rede Neural Convolucional (CNN)** para classificar imagens de condições climáticas. Utilizando o **Multi-class Weather Dataset**, o modelo é capaz de categorizar imagens em quatro classes distintas: **ensolarado**, **chuvoso**, **nublado** e **nevando**.

## Características Principais

- **Pré-processamento de Imagens**: Uso da classe `ImageDataGenerator` para normalização e aumento dos dados (*data augmentation*), incluindo `rescale`, `shear_range`, `zoom_range` e `horizontal_flip`. Separação dos dados em conjuntos de treinamento e validação na proporção de 80/20.

- **Arquitetura do Modelo**: Composto por camadas convolucionais e de pooling, seguido por camadas densas e `Dropout` para evitar overfitting.

- **Treinamento do Modelo**: Implementação de um callback personalizado para *early stopping* quando a acurácia de treinamento e validação ultrapassam 95%.

- **Visualização de Resultados**: Geração de gráficos de acurácia e perda para analisar o desempenho do modelo ao longo das épocas.

- **Predição em Novas Imagens**: Função para carregar uma imagem externa e prever sua classe climática usando o modelo treinado.

## Tecnologias Utilizadas

- **Python 3**
- **TensorFlow e Keras**
- **NumPy e Pandas**
- **Matplotlib e Seaborn**
- **PIL (Python Imaging Library)**

## Como Utilizar

1. **Clone o Repositório**:

   ```bash
   git clone https://github.com/augustompm/Multi-class-Weather-Classification.git

2. **Baixe e adicione o dataset no sub-diretório 'Multi-class Weather Dataset', que se encontra em:**

   ```url
   https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset?resource=download

Agradecimento: Adaptado a partir do livro Hands-On Guide To Image Classification de Vivian Siahaan.

Este projeto está licenciado sob Creative Commons.

