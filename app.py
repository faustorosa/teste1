import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
from PIL import Image
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Sistema de Predição de Obesidade",
    page_icon="🩺",
    layout="wide",
)

# --- FUNÇÕES DE EXIBIÇÃO DE PÁGINA ---

# Função para exibir o painel analítico
def show_dashboard():
    st.header("Painel Analítico: Insights sobre Fatores de Risco da Obesidade")
    st.markdown("""
    Esta seção apresenta uma análise visual dos dados utilizados para treinar nosso modelo. 
    Os insights podem ajudar a equipe médica a entender os principais fatores correlacionados 
    com os diferentes níveis de peso.
    """)

    # --- Insight 1 ---
    st.subheader("1. Impacto do Histórico Familiar")
    try:
        image = Image.open('3_historico_familiar_vs_obesidade.png')
        st.image(image, caption="Relação entre Histórico Familiar e Nível de Obesidade")
    except FileNotFoundError:
        st.warning("Imagem '3_historico_familiar_vs_obesidade.png' não encontrada.")

    st.markdown("""
    **Insight para a equipe médica:** Pacientes com histórico familiar de sobrepeso têm uma probabilidade drasticamente maior de desenvolver sobrepeso ou obesidade. A investigação do histórico familiar é um passo de triagem fundamental e de baixo custo.
    """)

    # --- Insight 2 ---
    st.subheader("2. Atividade Física como Fator de Proteção")
    try:
        image = Image.open('4_atividade_fisica_vs_obesidade.png')
        st.image(image, caption="Relação entre Frequência de Atividade Física e Nível de Obesidade")
    except FileNotFoundError:
        st.warning("Imagem '4_atividade_fisica_vs_obesidade.png' não encontrada.")
    st.markdown("""
    **Insight para a equipe médica:** A falta de atividade física está fortemente correlacionada com os níveis mais altos de obesidade. Incentivar a prática de exercícios (mesmo que 1-2 dias por semana) pode ser uma das intervenções mais eficazes.
    """)
    
    # --- Insight 3 ---
    st.subheader("3. O Transporte Diário Importa")
    try:
        image = Image.open('5_transporte_vs_obesidade.png')
        st.image(image, caption="Influência do Meio de Transporte no Nível de Obesidade")
    except FileNotFoundError:
        st.warning("Imagem '5_transporte_vs_obesidade.png' não encontrada.")
    st.markdown("""
    **Insight para a equipe médica:** O sedentarismo associado ao uso de Automóvel e Transporte Público é um fator de risco visível. Pacientes que utilizam esses meios podem precisar de atenção extra e incentivo a caminhadas ou outras atividades compensatórias.
    """)
    
    # --- Insight 4 ---
    st.subheader("4. Distribuição de Idade por Nível de Obesidade")
    try:
        image = Image.open('2_idade_vs_obesidade.png')
        st.image(image, caption="Relação entre Idade e Nível de Obesidade")
    except FileNotFoundError:
        st.warning("Imagem '2_idade_vs_obesidade.png' não encontrada.")
    st.markdown("""
    **Insight para a equipe médica:** A idade média tende a ser maior nos grupos com obesidade, sugerindo que o risco aumenta com o envelhecimento. Programas de prevenção podem ser focados em adultos jovens para evitar a progressão para a obesidade.
    """)

# Função para a página de predição
def show_predictor_header():
    st.markdown("<h1 style='text-align: left; color: #0077B6;'>🩺 Ferramenta de Apoio ao Diagnóstico de Obesidade</h1>", unsafe_allow_html=True)
    st.markdown("Preencha os dados na barra lateral à esquerda e clique no botão abaixo para realizar a predição.")

# --- INICIALIZAÇÃO DO SESSION STATE ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0
if 'errors' not in st.session_state:
    st.session_state.errors = {}
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False

# --- FUNÇÕES E CÁLCULOS EM CACHE ---
@st.cache_resource
def load_models_and_data():
    models = {}
    model_files = {
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl",
        "Regressão Logística": "regressao_logistica_model.pkl"
    }
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(filename)
        except FileNotFoundError:
            st.warning(f"Arquivo do modelo '{name}' ({filename}) não encontrado. Esta opção estará desabilitada.")
        except Exception as e:
            st.error(f"Erro ao carregar o modelo '{name}': {e}")
            
    try:
        df = pd.read_csv('Obesity.csv')
    except Exception as e:
        st.error(f"Erro fatal ao carregar o dataset 'Obesity.csv': {e}")
        st.stop()
        
    return models, df

def calculate_accuracy(_model, df):
    if 'Obesity' in df.columns and 'Obesity_level' not in df.columns:
        df = df.rename(columns={'Obesity': 'Obesity_level'})
    
    if 'TUE' in df.columns:
         df = df.rename(columns={'TUE': 'TER'})
         
    translation_map = {
        'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso Nível I',
        'Overweight_Level_II': 'Sobrepeso Nível II',
        'Obesity_Type_I': 'Obesidade Tipo I',
        'Obesity_Type_II': 'Obesidade Tipo II',
        'Obesity_Type_III': 'Obesidade Tipo III',
        'Insufficient_Weight': 'Peso Insuficiente'
    }
    df['Obesity_level'] = df['Obesity_level'].replace(translation_map)

    try:
        X = df.drop('Obesity_level', axis=1)
        y_true = df['Obesity_level']
        y_pred = _model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
    except Exception as e:
        st.warning(f"Não foi possível calcular a acurácia do modelo: {e}")
        return None

@st.cache_data
def get_model_insights_chart(model_name, _models):
    _model = _models[model_name]
    classifier = _model.named_steps['classifier']
    feature_names = _model.named_steps['preprocessor'].get_feature_names_out()
    df_importance = None

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        chart_title = 'Principais Fatores por Importância'
        x_axis_title = 'Nível de Importância'
        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})

    elif hasattr(classifier, 'coef_'):
        if len(classifier.coef_.shape) > 1:
            importances = np.abs(classifier.coef_[0])
        else:
            importances = np.abs(classifier.coef_)
        chart_title = 'Principais Fatores por Impacto'
        x_axis_title = 'Impacto no Modelo (Coeficiente Absoluto)'
        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    
    if df_importance is None:
        return None

    df_importance = df_importance.sort_values('importance', ascending=False).head(10)

    translation_dict = {
        'num__Age': 'Idade', 'num__Height': 'Altura', 'num__Weight': 'Peso',
        'num__FCVC': 'Consumo de Vegetais', 'num__NCP': 'Nº de Refeições Principais',
        'num__CH2O': 'Consumo de Água', 'num__FAF': 'Atividade Física',
        'num__TER': 'Tempo de Uso de Telas', 'cat__Gender_Male': 'Gênero (Masculino)',
        'cat__family_history_yes': 'Histórico Familiar de Sobrepeso',
        'cat__FAVC_yes': 'Consome Alta Caloria (FAVC)', 'cat__CAEC_Sometimes': 'Lanches (Às vezes)',
        'cat__SCC_yes': 'Monitora Calorias', 'cat__SMOKE_yes': 'Fumante',
        'cat__CALC_Sometimes': 'Consumo de Álcool (Às vezes)',
        'cat__MTRANS_Public_Transportation': 'Transporte Público',
    }
    df_importance['feature_translated'] = df_importance['feature'].apply(
        lambda x: translation_dict.get(x, x.replace('num__', '').replace('cat__', '').replace('_', ' ').title())
    )

    chart = alt.Chart(df_importance).mark_bar(opacity=0.8, color="#0077B6").encode(
        x=alt.X('importance:Q', title=x_axis_title),
        y=alt.Y('feature_translated:N', sort='-x', title='Característica'),
        tooltip=[alt.Tooltip('feature_translated', title='Característica'), alt.Tooltip('importance', title='Importância', format='.4f')]
    ).properties(
        title=chart_title
    )
    return chart

# --- LÓGICA DE CONTROLE DE ESTADO ---
def reset_app():
    st.session_state.prediction_result = None
    st.session_state.edit_mode = False
    st.session_state.errors = {}
    st.session_state.reset_counter += 1

def enable_edit_mode():
    st.session_state.edit_mode = True
    st.session_state.prediction_result = None

def render_input(widget_type, label, options, key, **kwargs):
    dynamic_key = f"{key}_{st.session_state.reset_counter}"
    
    if st.session_state.get(dynamic_key) is not None and dynamic_key in st.session_state.errors:
        del st.session_state.errors[dynamic_key]

    if widget_type == 'selectbox':
        input_widget = st.selectbox(label, options, key=dynamic_key, **kwargs)
    elif widget_type == 'radio':
        input_widget = st.radio(label, options, key=dynamic_key, **kwargs)
    
    if dynamic_key in st.session_state.errors:
        error_message_style = "<p style='font-size: 12px; color: red; margin-top: -15px; margin-bottom: 5px;'>Campo obrigatório</p>"
        st.markdown(error_message_style, unsafe_allow_html=True)
        
    return input_widget

# --- NAVEGAÇÃO ---
with st.sidebar.container(border=True):
    st.subheader("Navegação")  # Movido para dentro do container
    app_mode = st.radio(
        "Escolha a funcionalidade:",
        ["Painel Analítico", "Sistema Preditivo"],
        horizontal=True,
        label_visibility="collapsed"
    )

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if app_mode == "Sistema Preditivo":
    
    models, df = load_models_and_data()
    
    if not models:
        st.error("Nenhum modelo de predição foi carregado. A aplicação não pode continuar. Por favor, execute o notebook de treinamento e coloque os arquivos .pkl no diretório correto.")
        st.stop()
    
    gender_map = {'Feminino': 'Female', 'Masculino': 'Male'}
    yes_no_map = {'Sim': 'yes', 'Não': 'no'}
    caec_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
    calc_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently'}
    mtrans_map = {'Transporte Público': 'Public_Transportation', 'Automóvel': 'Automobile', 'Caminhando': 'Walking', 'Moto': 'Motorbike', 'Bicicleta': 'Bike'}

    with st.sidebar:
        is_disabled = (st.session_state.prediction_result is not None) and not st.session_state.edit_mode
        reset_key = st.session_state.reset_counter
        st.header("Configurações da Predição")

        model_names_original = list(models.keys())
        model_names_display = [f"{name} (default)" if name == "SVM" else name for name in model_names_original]
        try:
            default_index = model_names_display.index("SVM (default)")
        except ValueError:
            default_index = 0

        model_selection_display = st.selectbox(
            "Escolha o modelo de classificação:",
            model_names_display,
            index=default_index,
            disabled=is_disabled,
            key=f'model_selection_{reset_key}' 
        )
        
        model_selection = model_selection_display.replace(" (default)", "")
        
        active_model = models[model_selection]
        
        st.divider()

        st.header("Insira os Dados para Análise")
        
        st.subheader("Dados Demográficos")
        age = st.number_input('Idade', min_value=1, max_value=100, value=None, placeholder="Exemplo: 25", disabled=is_disabled, key=f'age_{reset_key}')
        height = st.number_input('Altura (m)', min_value=1.0, max_value=2.5, value=None, placeholder="Exemplo: 1,70", format="%.2f", disabled=is_disabled, key=f'height_{reset_key}')
        weight = st.number_input('Peso (kg)', min_value=30.0, max_value=200.0, value=None, placeholder="Exemplo: 70,0", format="%.1f", disabled=is_disabled, key=f'weight_{reset_key}')
        
        gender_label = render_input('radio', 'Gênero', list(gender_map.keys()), key='gender_input', index=None, horizontal=True, disabled=is_disabled)
        
        st.subheader("Histórico e Hábitos")
        family_history_label = render_input('radio', 'Histórico Familiar de Sobrepeso?', ['Sim', 'Não'], key='family_history_input', horizontal=True, index=None, disabled=is_disabled)
        favc_label = render_input('radio', 'Consome alta caloria (FAVC)?', ['Sim', 'Não'], key='favc_input', horizontal=True, index=None, disabled=is_disabled)
        scc_label = render_input('radio', 'Monitora calorias?', ['Sim', 'Não'], key='scc_input', horizontal=True, index=None, disabled=is_disabled)
        smoke_label = render_input('radio', 'Fumante?', ['Sim', 'Não'], key='smoke_input', horizontal=True, index=None, disabled=is_disabled)
            
        caec_label = render_input('selectbox', 'Lanches entre refeições (CAEC)?', list(caec_map.keys()), key='caec_input', index=None, placeholder="Selecione...", disabled=is_disabled)
        calc_label = render_input('selectbox', 'Consumo de álcool (CALC)?', list(calc_map.keys()), key='calc_input', index=None, placeholder="Selecione...", disabled=is_disabled)
        mtrans_label = render_input('selectbox', 'Principal Transporte (MTRANS)?', list(mtrans_map.keys()), key='mtrans_input', index=None, placeholder="Selecione...", disabled=is_disabled)
        
        st.subheader("Rotina Diária")
        fcvc = st.slider('Frequência de consumo de vegetais (1-3)?', 1, 3, 1, disabled=is_disabled, key=f'fcvc_{reset_key}')
        ncp = st.slider('Nº de refeições principais (1-4)?', 1, 4, 1, disabled=is_disabled, key=f'ncp_{reset_key}')
        ch2o = st.slider('Consumo de água - litros/dia (1-3)?', 1, 3, 1, disabled=is_disabled, key=f'ch2o_{reset_key}')
        faf = st.slider('Atividade física - dias/semana (0-3)?', 0, 3, 0, disabled=is_disabled, key=f'faf_{reset_key}')
        tue = st.slider('Tempo de uso de telas - horas/dia (0-2)?', 0, 2, 0, disabled=is_disabled, key=f'tue_{reset_key}')
    
    # --- ÁREA PRINCIPAL ---
        accuracy = calculate_accuracy(active_model, df.copy())
    model_insights_chart = get_model_insights_chart(model_selection, models)

    st.markdown("<h1 style='text-align: left; color: #0077B6;'>🩺 Ferramenta de Apoio ao Diagnóstico de Obesidade</h1>", unsafe_allow_html=True)
    
    sub_header_col, metric_col = st.columns([4, 1])
    with sub_header_col:
        st.markdown("Preencha os dados na barra lateral à esquerda e clique no botão abaixo para realizar a predição.")
    with metric_col:
        if accuracy is not None:
            st.metric(label=f"Acurácia ({model_selection})", value=f"{accuracy*100:.2f}%", help="Mede a porcentagem de previsões corretas do modelo em todo o conjunto de dados.")

    st.markdown("---")
    
    if st.session_state.errors:
        st.warning("⚠️ Por favor, preencha todos os campos obrigatórios.")
    
    button_placeholder = st.empty()

    if st.session_state.prediction_result is None or st.session_state.edit_mode:
        if button_placeholder.button('**Realizar Predição**', type="primary", use_container_width=True):
            st.session_state.errors = {}
            
            current_reset_key = st.session_state.reset_counter
            inputs_to_validate = {
                'age': age, 'height': height, 'weight': weight,
                f'gender_input_{current_reset_key}': gender_label, 
                f'family_history_input_{current_reset_key}': family_history_label,
                f'favc_input_{current_reset_key}': favc_label, 
                f'scc_input_{current_reset_key}': scc_label, 
                f'smoke_input_{current_reset_key}': smoke_label, 
                f'caec_input_{current_reset_key}': caec_label, 
                f'calc_input_{current_reset_key}': calc_label, 
                f'mtrans_input_{current_reset_key}': mtrans_label
            }
            
            errors = {key: True for key, value in inputs_to_validate.items() if value is None}

            if errors:
                st.session_state.errors = {k: v for k, v in errors.items() if not k in ['age', 'height', 'weight']}
                st.rerun()
            else:
                st.session_state.edit_mode = False
                
                input_values = {
                    'Gender': gender_map[gender_label], 'Age': age, 'Height': height, 'Weight': weight,
                    'family_history': yes_no_map[family_history_label], 
                    'FAVC': yes_no_map[favc_label],
                    'FCVC': float(fcvc), 'NCP': float(ncp), 'CAEC': caec_map[caec_label],
                    'SMOKE': yes_no_map[smoke_label], 'CH2O': float(ch2o), 'SCC': yes_no_map[scc_label],
                    'FAF': float(faf), 'TER': float(tue), 'CALC': calc_map[calc_label],
                    'MTRANS': mtrans_map[mtrans_label]
                }
                
                input_data = pd.DataFrame([input_values])

                with st.spinner(f'Analisando os dados com o modelo {model_selection}...'):
                    prediction = active_model.predict(input_data)
                    prediction_proba = active_model.predict_proba(input_data)
                    
                    report_values = {
                        'family_history': yes_no_map[family_history_label], 'favc': yes_no_map[favc_label],
                        'fcvc': fcvc, 'ncp': ncp, 'caec': caec_map[caec_label], 'smoke': yes_no_map[smoke_label],
                        'ch2o': ch2o, 'scc': yes_no_map[scc_label], 'faf': faf, 'tue': tue,
                        'mtrans': mtrans_map[mtrans_label]
                    }
                    st.session_state.prediction_result = (prediction, prediction_proba, report_values, model_selection)
                    st.rerun()
    else:
        col1_btn, col2_btn = button_placeholder.columns(2)
        col1_btn.button('**⬅️ Realizar Nova Predição**', use_container_width=True, on_click=reset_app)
        col2_btn.button('**📝 Editar Dados Informados**', use_container_width=True, on_click=enable_edit_mode)

    if st.session_state.prediction_result is not None:
        prediction, prediction_proba, input_values, used_model = st.session_state.prediction_result
        
        st.markdown(f"<h2 style='text-align: center;'>Resultado da Predição (Modelo: {used_model})</h2>", unsafe_allow_html=True)
        
        color_map_by_translation = {
            'Peso Normal': '#2ECC71', 'Sobrepeso Nível I': '#F1C40F',
            'Sobrepeso Nível II': '#E67E22', 'Obesidade Tipo I': '#E74C3C',
            'Obesidade Tipo II': '#C0392B', 'Obesidade Tipo III': '#A93226',
            'Peso Insuficiente': '#3498DB'
        }

        result_translation = prediction[0]
        result_color = color_map_by_translation.get(result_translation, '#34495E')

        st.markdown(f"<h2 style='text-align: center; color: {result_color};'>{result_translation}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confiança do modelo no resultado: <strong>{np.max(prediction_proba)*100:.2f}%</strong>.</p>", unsafe_allow_html=True)
        
        _, center_col, _ = st.columns([0.5, 3, 0.5])
        
        with center_col:
            st.markdown("""
            <style>
            div[data-testid="stExpander"] summary {
                position: relative;
                background-color: #0077B6;
                color: white;
                border-radius: 0.25rem;
            }
            div[data-testid="stExpander"] summary p {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 18px;
                font-weight: 600;
                width: 90%;
                text-align: center;
            }
            div[data-testid="stExpander"] summary svg {
                fill: white;
            }
            </style>
            """, unsafe_allow_html=True)

            with st.expander("🔎 Clique para ver a análise detalhada dos seus hábitos"):
                st.markdown("<h4 style='text-align: center;'>Análise de Hábitos</h4>", unsafe_allow_html=True)
                
                risk_factors, protective_factors = [], []
                if input_values['family_history'] == 'yes': risk_factors.append("Possui histórico familiar de sobrepeso.")
                if input_values['favc'] == 'yes': risk_factors.append("Consome alimentos de alta caloria frequentemente.")
                else: protective_factors.append("Não consome alimentos de alta caloria frequentemente.")
                if input_values['fcvc'] < 2: risk_factors.append("Baixo consumo de vegetais.")
                else: protective_factors.append("Bom consumo de vegetais.")
                if input_values['ncp'] < 3: risk_factors.append("Realiza menos de 3 refeições principais.")
                else: protective_factors.append("Realiza 3 ou mais refeições principais.")
                if input_values['caec'] in ['Frequently', 'Always']: risk_factors.append("Lancha com muita frequência.")
                if input_values['smoke'] == 'yes': risk_factors.append("É fumante.")
                else: protective_factors.append("Não é fumante.")
                if input_values['ch2o'] < 2: risk_factors.append("Bebe menos de 2 litros de água por dia.")
                else: protective_factors.append("Bom consumo de água.")
                if input_values['scc'] == 'yes': protective_factors.append("Monitora o consumo de calorias.")
                else: risk_factors.append("Não monitora as calorias que consome.")
                if input_values['faf'] < 2: risk_factors.append("Pratica pouca atividade física.")
                else: protective_factors.append("Pratica atividade física regularmente.")
                if input_values['tue'] > 1: risk_factors.append("Passa muito tempo em frente a telas.")
                else: protective_factors.append("Uso moderado de telas.")
                if input_values['mtrans'] in ['Automobile', 'Public_Transportation']: risk_factors.append("Usa transporte que incentiva o sedentarismo.")
                elif input_values['mtrans'] in ['Walking', 'Bike']: protective_factors.append("Usa transporte ativo (caminhada, bicicleta).")

                col_risk, col_prot = st.columns(2)
                with col_risk:
                    st.markdown("<h5><span style='color: #E74C3C;'>🔴 Fatores de Risco</span></h5>", unsafe_allow_html=True)
                    if risk_factors:
                        for factor in risk_factors: st.markdown(f"- {factor}")
                    else: st.markdown("- Nenhum fator de risco óbvio identificado.")
                with col_prot:
                    st.markdown("<h5><span style='color: #2ECC71;'>🟢 Fatores Protetivos</span></h5>", unsafe_allow_html=True)
                    if protective_factors:
                        for factor in protective_factors: st.markdown(f"- {factor}")
                    else: st.markdown("- Nenhum fator de proteção óbvio identificado.")

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: center;'>Probabilidade Detalhada por Classe</h4>", unsafe_allow_html=True)

                df_proba = pd.DataFrame(prediction_proba, columns=active_model.classes_).T.reset_index()
                df_proba.columns = ['Classe Original', 'Probabilidade']
                df_proba['Classe'] = df_proba['Classe Original']
                df_proba['Probabilidade_num'] = df_proba['Probabilidade']
                df_proba = df_proba.sort_values(by='Probabilidade_num', ascending=False)
                df_proba['Probabilidade'] = df_proba['Probabilidade_num'].apply(lambda p: f"{p*100:.2f}%")
                
                st.dataframe(df_proba[['Classe', 'Probabilidade']], use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if model_insights_chart:
                with st.popover(f"Ver Análise de Fatores do Modelo ({used_model})", use_container_width=True):
                    st.altair_chart(model_insights_chart, use_container_width=True)
                    st.info("Este gráfico mostra os fatores que o modelo mais considera. Para modelos como Regressão Logística, a análise é baseada no valor absoluto dos coeficientes.")
            elif used_model == "SVM":
                st.info("ℹ️ A análise de fatores não está disponível para o modelo SVM, pois ele utiliza um kernel não-linear. Este tipo de modelo não possui uma lista direta de coeficientes de importância como os modelos lineares ou baseados em árvores.")

elif app_mode == "Painel Analítico":
    show_dashboard()