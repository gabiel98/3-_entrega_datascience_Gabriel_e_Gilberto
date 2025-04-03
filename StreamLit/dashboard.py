import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
import base64

# Configurações da página
st.set_page_config(page_title="Análise Pré-Natal 2019", page_icon="👶", layout="wide", initial_sidebar_state="collapsed")
st.title("Análise da Qualidade do Pré-Natal - Brasil 2019")

def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

# Definindo a URL da imagem de fundo
image_url = get_image_base64(r'C:/Users/Alunos/Downloads/dash/StreamLit/fundo_dash.jpg')  # URL da sua imagem

# Alterando a configuração de temas para não permitir que o usuário modifique o tema
st.markdown(f"""
    <style>
        /* Aplicando a imagem de fundo a toda a página, incluindo o conteúdo e a barra lateral */
        body {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)), url('{image_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            margin: 0;
            font-size: 12px !important;  /* Tamanho base para todo o texto */
        }}
        
        /* Garantindo que o conteúdo da página seja visível e não sobreponha o fundo */
        .reportview-container {{
            background-color: rgba(255, 255, 255, 0.8);  /* Fundo translúcido para o conteúdo */
            padding: 2rem;
            min-height: 100vh; /* Para garantir que o conteúdo ocupe toda a tela */
        }}

        /* Escondendo a barra lateral */
        .css-1lcb2n6 {{
            display: none;  /* Esconde a barra lateral */
        }}

        /* Escondendo a barra de navegação e outras decorações do Streamlit */
        #MainMenu {{
            visibility: hidden;  /* Esconde o menu principal */
        }}
        .stDeployButton {{
            display: none;  /* Esconde o botão de Deploy */
        }}
        footer {{
            visibility: hidden;  /* Esconde o rodapé */
        }}
        #stDecoration {{
            display: none;  /* Esconde decorações extras */
        }}
        .stApp, .stAppHeader {{
         background: none;
        }}
        p {{ font-size: 1.2rem !important; }}
        span > div {{ font-size: 1.4rem !important; }}
        
    </style>

    </style>
""", unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_data():
    # Ajuste o caminho conforme necessário
    return pd.read_parquet(r'C:\Users\Alunos\Downloads\dash\StreamLit\pns2019.parquet')

df = load_data()

# Cria abas para as três seções
tab1, tab2, tab3 = st.tabs(["Estatísticas Descritivas", "Gráficos Interativos", "Modelo de IA"])

# ================================================
# Seção 1: Estatísticas Descritivas
# ================================================
with tab1:
    st.header("📈 Estatísticas Descritivas")

    # Estatística 1 - Moda das Semanas
    with st.expander("1. Semana mais comum para 1ª consulta", expanded=True):
        valid_data = df['S06901'].loc[df['S06901'].between(1, 41)]
        moda = valid_data.mode().values[0]
        st.metric(label="Semana da Primeira Consulta", value=f"{int(moda)}ª semana")

    # Estatística 2 - Saída do Hospital
    with st.expander("2. Saída do Hospital com o Bebê", expanded=True):
        valid_data = df['S132'].dropna()
        total = len(valid_data)
        sim = (valid_data == 1).sum()
        nao = (valid_data == 2).sum()
        
        col1, col2 = st.columns(2)
        col1.metric("Saíram com o bebê (Sim)", f"{sim} mulheres", f"{(sim/total)*100:.2f}%")
        col2.metric("Não saíram com o bebê (Não)", f"{nao} mulheres", f"{(nao/total)*100:.2f}%")

    # Estatística 3 - Score de Exames Realizados
    with st.expander("3. Score de Exames Realizados", expanded=True):
        st.write("⭐ Pontuação dada de acordo com a quantidade de exames realizados pelas gestantes")
        colunas_score = ['S07901', 'S07902', 'S07903', 'S07904', 'S07905', 'S088', 'S090', 'S095', 'S080']
        df_score = df[colunas_score].copy()
        
        # Cálculo do score: cada exame com valor 1 conta como 1, caso contrário 0
        df_score = df_score.applymap(lambda x: 1 if x == 1 else 0)
        df_score['Score_Final'] = df_score.sum(axis=1)
        
        # Tabela de resumo
        score_counts = df_score['Score_Final'].value_counts().sort_index()
        score_percent = (score_counts / len(df_score) * 100).round(2)
        
        st.dataframe(
            pd.DataFrame({
                'Score': score_counts.index,
                'Mulheres': score_counts.values,
                'Percentual (%)': score_percent.values
            }),
            hide_index=True,
            use_container_width=True
    
        )



# ================================================
# Seção 2: Gráficos Interativos (Tamanhos Ajustados)
# ================================================
with tab2:
    st.header("📊 Visualizações Interativas")

    # Gráfico 1 - Distribuição de Mulheres que Fizeram Pré-Natal
    with st.expander("Distribuição de Mulheres que Fizeram Pré-Natal"):
        counts = df['S068'].value_counts()
        labels = ['Sim' if i == 1 else 'Não' if i == 2 else 'Outros' for i in counts.index]

        fig1 = px.pie(
            values=counts,
            names=labels,
            color_discrete_sequence=['#FFB6C1', '#B0E0E6', '#D3D3D3'],
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2 - Complicações Durante/Após o Parto
    with st.expander("Complicações Durante/Após o Parto"):
        counts_parto = df['S125'].map({1: 'Sim', 2: 'Não'}).value_counts()
        counts_pos = df['S126'].map({1: 'Sim', 2: 'Não'}).value_counts()

        df_complicacoes = pd.DataFrame({
            "Categoria": ["Durante", "Após"],
            "Sim": [counts_parto.get("Sim", 0), counts_pos.get("Sim", 0)],
            "Não": [counts_parto.get("Não", 0), counts_pos.get("Não", 0)]
        })

        fig2 = px.bar(
            df_complicacoes,
            x="Categoria",
            y=["Sim", "Não"],
            barmode="stack",
            labels={'variable': 'Complicações', 'value': 'Valores'},
            color_discrete_sequence=['#FFB6C1', '#B0E0E6'],
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3 - Relação Número de Consultas e Problemas
    with st.expander("Relação Número de Consultas e Problemas"):
        df_teste = df.copy()
        df_teste['consultas'] = df['S070'].replace({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7})
        df_teste.loc[df['S068'] == 2, 'consultas'] = 0
        df_teste['Teve Problemas'] = df[['S125', 'S126']].apply(
            lambda x: 'Teve problemas' if 1 in x.values else 'Não teve problemas', axis=1
        )

        mapeamento = {0: 'Não fez', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7+'}
        df_teste['Consultas'] = df_teste['consultas'].map(mapeamento)
        contagem = df_teste.groupby(['Consultas', 'Teve Problemas']).size().unstack()

        fig3 = px.bar(
            contagem,
            x=contagem.index,
            y=contagem.columns,
            labels={'value': 'Valores'},
            color_discrete_sequence=['#FFB6C1', '#B0E0E6'],
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

# ================================================
# Seção 3: Modelo de IA
# ================================================

with tab3:
    st.markdown("""
    <style>
        /* Container principal */
        div[data-testid="stNumberInputContainer"] {
            width: 160px !important;
            min-width: 160px !important;
            max-width: 140px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Div interna com o input */
        .st-emotion-cache-1yiu4b3 > div[data-baseweb="input"] {
            width: 80px !important;
            min-width: 80px !important;
            padding: 0 !important;
        }

        /* Campo de entrada numérico */
        input[data-testid="stNumberInputField"] {
            width: 60px !important;
            height: 100px !important;
            padding: 0 8px !important;
            font-size: 20px !important;
            min-width: 60px !important;
        }

        /* Container dos botões (+/-) */
        .st-emotion-cache-1k5fi8b {
            width: 80px !important;
            min-width: 80px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Botões individualmente */
        button.st-emotion-cache-ml50x9 {
            width: 40px !important;
            height: 40px !important;
            min-width: 35px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Ícones dentro dos botões */
        .st-emotion-cache-1ejyxxe,
        .st-emotion-cache-14zer8g {
            width: 10px !important;
            height: 10px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Filtrando e processando os dados conforme o código fornecido
    colunas_desejadas = [
        'S068',  # Quando estava grávida fez alguma consulta de pré-natal?
        'S07901', # Consultas que mediram pressão arterial
        'S07902', # Consultas que mediram peso
        'S07903', # Consultas que mediram barriga
        'S07904', # Consultas que ouviram o coração do bebê
        'S07905', # Consultas que examinaram as mamas
        'S080',  # Teste para sífilis
        'S082', # Resultado do teste de sífilis
        'S11801', # Semanas de gravidez no momento do parto
        'S125',  # Complicações durante o parto
        'S126'   # Complicações após o parto
    ]

    # Filtrando o DataFrame para incluir apenas as colunas desejadas
    df_filtrado = df[colunas_desejadas]

    # Transformando as colunas de respostas binárias (1 = sim, 2 = não), ignorando valores fora de 1 e 2
    cols_binarias = ['S068', 'S125', 'S126']
    df_filtrado[cols_binarias] = df_filtrado[cols_binarias].applymap(lambda x: x if x in [1, 2] else np.nan)

    # Alterando as colunas de respostas para consultas
    consulta_teste = ['S080']
    df_filtrado.loc[df_filtrado['S068'] == 2, consulta_teste] = df_filtrado.loc[df_filtrado['S068'] == 2, consulta_teste].applymap(lambda x: x if x in [1, 2] else 2)

    # Alterando as colunas de respostas para consultas
    cols_consultas = ['S082','S07901', 'S07902', 'S07903', 'S07904', 'S07905']
    df_filtrado.loc[df_filtrado['S068'] == 2, cols_consultas] = df_filtrado.loc[df_filtrado['S068'] == 2, cols_consultas].applymap(lambda x: x if x in [1, 2, 3] else 3)

    # Criando a coluna 'prematuro' com base em S11801
    df_filtrado['prematuro'] = df_filtrado['S11801'].apply(lambda x: 1 if x < 37 else (2 if x >= 37 else np.nan))

    # Dropar a coluna 'S11801' do DataFrame
    df_filtrado = df_filtrado.drop('S11801', axis=1)

    # Definindo o modelo de árvore de decisão e parâmetros para otimização
    model_1 = DecisionTreeClassifier(random_state=42)
    param_grid_1 = {
        'criterion': ['entropy'],
        'max_depth': [6],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
        'max_features': ['log2'],
        'class_weight': ['balanced'],
        'ccp_alpha': [0.001]
    }

    # Aplicando Random Oversampling para balanceamento das classes
    X1_train = df_filtrado.drop(columns=['S125', 'S126'])  # Variáveis independentes
    y1_train = df_filtrado['S125']  # Variável dependente (complicações durante o parto)

    ros = RandomOverSampler(random_state=42)
    X1_res, y1_res = ros.fit_resample(X1_train, y1_train)

    # GridSearchCV para otimização de parâmetros
    grid_search_1 = GridSearchCV(estimator=model_1, param_grid=param_grid_1, cv=5, n_jobs=-1, verbose=2)

    # Treinar o modelo com GridSearchCV
    grid_search_1.fit(X1_res, y1_res)

    # Melhor modelo encontrado
    best_model_1 = grid_search_1.best_estimator_

    # Novo modelo para complicações APÓS o parto
    X2_train = df_filtrado.drop(columns=['S125', 'S126'])
    y2_train = df_filtrado['S126']
    X2_res, y2_res = ros.fit_resample(X2_train, y2_train)

    # Parâmetros do segundo modelo
    model_2 = DecisionTreeClassifier(random_state=42)
    param_grid_2 = {
        'criterion': ['gini'],
        'max_depth': [7],
        'min_samples_split': [10],
        'min_samples_leaf': [11],
        'max_features': ['log2'],
        'class_weight': ['balanced']
    }

    grid_search_2 = GridSearchCV(model_2, param_grid_2, cv=5, n_jobs=-1, verbose=2)
    grid_search_2.fit(X2_res, y2_res)
    best_model_2 = grid_search_2.best_estimator_


    # =========================================
    # Interação com o Usuário via Streamlit
    # =========================================
    st.header("🤖 Estimar risco de complicações durante e após parto")

    # Perguntas de resposta binária (Sim/Não) - Usando botões
    st.write("Escolha a resposta para as seguintes perguntas:")

    # Inicializar variáveis para armazenar as respostas do usuário
    user_respostas = {
        'S068': None,
        'S07901': None,
        'S07902': None,
        'S07903': None,
        'S07904': None,
        'S07905': None,
        'S080': None,
        'S082': None,
        'prematuro': None
    }

    col1, col2 = st.columns(2)
    with col1:   
        # Pergunta 1: Quando estava grávida fez alguma consulta de pré-natal?
        resposta = st.radio("Quando estava grávida, fez alguma consulta de pré-natal?", options=["Sim", "Não"])
        user_respostas['S068'] = 1 if resposta == "Sim" else 2

        # Pergunta 4: Quantas consultas foram realizadas para medir a pressão arterial durante a gravidez?
        num_consultas = st.number_input("Quantas consultas foram realizadas para medir a pressão arterial?", min_value=0, max_value=50, step=1)
        user_respostas['S07901'] = 1 if 1 < num_consultas < 50 else (2 if num_consultas > 0 else 3)

        # Pergunta 5: Quantas consultas foram realizadas para medir o peso durante a gravidez?
        num_consultas = st.number_input("Quantas consultas foram realizadas para medir o peso?", min_value=0, max_value=50, step=1)
        user_respostas['S07902'] = 1 if 7 < num_consultas < 50 else (2 if num_consultas > 0 else 3)

        # Pergunta 6: Quantas consultas foram realizadas para medir a barriga durante a gravidez?
        num_consultas = st.number_input("Quantas consultas foram realizadas para medir a barriga?", min_value=0, max_value=50, step=1)
        user_respostas['S07903'] = 1 if 7 < num_consultas < 50 else (2 if num_consultas > 0 else 3)
    
        # Pergunta 7: Quantas consultas foram realizadas para ouvir o coração do bebê?
        num_consultas = st.number_input("Quantas consultas foram realizadas para ouvir o coração do bebê?", min_value=0, max_value=50, step=1)
        user_respostas['S07904'] = 1 if 7 < num_consultas < 50 else (2 if num_consultas > 0 else 3)
   
    with col2: 
        # Pergunta 8: Quantas consultas foram realizadas para examinar as mamas durante a gravidez?
        num_consultas = st.number_input("Quantas consultas foram realizadas para examinar as mamas?", min_value=0, max_value=50, step=1)
        user_respostas['S07905'] = 1 if 7 < num_consultas < 50 else (2 if num_consultas > 0 else 3)

        # Pergunta 9: Foi realizado o teste para sífilis durante a gravidez?
        resposta = st.radio("Foi realizado o teste para sífilis durante a gravidez?", options=["Sim", "Não"])
        user_respostas['S080'] = 1 if resposta == "Sim" else 2

        if user_respostas['S080'] == 1:
            resposta = st.radio("Qual o resultado do teste para sífilis?", options=["Positivo", "Negativo"])
            user_respostas['S082'] = 1 if resposta == "Positivo" else 2
        else:
            user_respostas['S082'] = 3


         # Pergunta 10: Quantas semanas de gestação você tinha no momento do parto?
        semanas_gestacao = st.number_input("Quantas semanas de gestação você tinha no momento do parto?", min_value=1, max_value=45, step=1)
        user_respostas['prematuro'] = 1 if semanas_gestacao < 37 else 2

    
    # =========================================
    # Exemplo de Previsão de Risco com o Modelo
    # =========================================

    # Criar uma variável com os dados de entrada para previsão (não alterando o DataFrame original)
    input_data = pd.DataFrame({
        'S068': [user_respostas['S068']],
        'S07901': [user_respostas['S07901']],  # Consultas que mediram pressão arterial
        'S07902': [user_respostas['S07902']],  # Consultas que mediram peso
        'S07903': [user_respostas['S07903']],  # Consultas que mediram barriga
        'S07904': [user_respostas['S07904']],  # Consultas que ouviram o coração do bebê
        'S07905': [user_respostas['S07905']],  # Consultas que examinaram as mamas
        'S080': [user_respostas['S080']],  # Teste sífilis
        'S082': [user_respostas['S082']], # Resultado do teste de sífilis
        'prematuro': [user_respostas['prematuro']]  # Se prematuro ou não
    })

    # Botão para calcular o risco
    if st.button("Calcular Risco"):
        risco = best_model_1.predict(input_data)
        chance = np.max(best_model_1.predict_proba(input_data)) * 100
        risco2 = best_model_2.predict(input_data)
        chance2 = np.max(best_model_2.predict_proba(input_data)) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Risco de complicações durante o parto: {'🔴Sim' if risco[0] == 1 else '🟢Não'}<br>"
            f"Com chance de: {chance:.2f}%", 
            unsafe_allow_html=True)
        
        with col2:
            st.write(f"Risco de complicações após o parto: {'🔴Sim' if risco2[0] == 1 else '🟢Não'}<br>"
            f"Com chance de: {chance2:.2f}%", 
            unsafe_allow_html=True
            )
    # Disclaimer
    st.warning("""
    **Aviso Importante:**  
    Este modelo está em fase beta.  
    Tem fins educacionais e não substitui avaliação profissional.  
    Dados baseados na PNS 2019 - IBGE.
    """)