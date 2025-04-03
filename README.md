# Tutorial

# Guia de Instalação e Configuração

## 1. Instalar as Dependências

pip install streamlit
pip install pandas
pip install plotly
pip install numpy
pip install scikit-learn
pip install imbalanced-learn

## 2. Criar Arquivo para Estilo
Crie o arquivo de configuração para o Streamlit no seguinte caminho:

C:\Users\SeuUsuario\.streamlit\config.toml

Dentro deste arquivo, cole o código abaixo:

toml
Copiar
[server]
headless = true  

[theme]
base = "dark" 
primaryColor = "#F63366"
backgroundColor = "#f6f0f0"
secondaryBackgroundColor = "#f5e4e4"
textColor = "#302626"

## 3. Atenção para os Caminhos dos Arquivos
Não se esqueça de alterar os caminhos dos arquivos no código:

Linha 22: Caminho do arquivo fundo_dash.jpg

Linha 78: Caminho do arquivo pns2019.parquet

Certifique-se de que os arquivos estão nos locais corretos para que a aplicação funcione corretamente.
