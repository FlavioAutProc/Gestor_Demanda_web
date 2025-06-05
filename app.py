import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from io import BytesIO
from fpdf import FPDF
import warnings



warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Configuração da página
st.set_page_config(
    page_title="Sistema de Previsão de Demanda - Padaria Master",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)


class DemandForecastSystem:
    def __init__(self):
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas'])
        self.forecast_results = None
        self.initialize_session_state()

    def initialize_session_state(self):
        """Inicializa o estado da sessão para persistência de dados"""
        if 'data' not in st.session_state:
            st.session_state.data = self.data
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {
                'ARIMA': {'order': (5, 1, 0)},
                'Holt-Winters': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
                'Random Forest': {'n_estimators': 100, 'max_depth': None}
            }

    def load_data(self):
        """Carrega dados da sessão"""
        self.data = st.session_state.data
        self.forecast_results = st.session_state.forecast_results

    def save_data(self):
        """Salva dados na sessão"""
        st.session_state.data = self.data
        st.session_state.forecast_results = self.forecast_results

    def import_data(self, uploaded_file):
        """Importa dados de arquivo"""
        try:
            if uploaded_file.name.endswith('.xlsx'):
                new_data = pd.read_excel(uploaded_file)
            else:
                new_data = pd.read_csv(uploaded_file)

            # Verificar colunas necessárias
            if 'Data' not in new_data.columns or 'Unidades Vendidas' not in new_data.columns:
                st.error("O arquivo deve conter colunas 'Data' e 'Unidades Vendidas'")
                return False

            # Converter data para datetime
            new_data['Data'] = pd.to_datetime(new_data['Data'])

            # Ordenar por data
            new_data = new_data.sort_values('Data')

            # Verificar duplicatas
            if new_data['Data'].duplicated().any():
                st.warning("Foram encontradas datas duplicadas. Serão mantidos apenas os últimos valores.")
                new_data = new_data.drop_duplicates('Data', keep='last')

            self.data = new_data
            self.save_data()
            st.success(f"Dados importados com sucesso! {len(self.data)} registros carregados.")
            return True

        except Exception as e:
            st.error(f"Falha ao importar arquivo: {str(e)}")
            return False

    def add_manual_data(self, date, units):
        """Adiciona dados inseridos manualmente"""
        try:
            date = pd.to_datetime(date)
            units = float(units)

            # Verificar se data já existe
            if not self.data.empty and date in pd.to_datetime(self.data['Data']).values:
                # Remover entrada existente
                self.data = self.data[pd.to_datetime(self.data['Data']) != date]

            # Adicionar novo dado
            new_data = pd.DataFrame([[date, units]], columns=['Data', 'Unidades Vendidas'])
            self.data = pd.concat([self.data, new_data], ignore_index=True)
            self.data.sort_values('Data', inplace=True)
            self.save_data()
            st.success("Dados adicionados com sucesso!")
            return True

        except ValueError as e:
            st.error(f"Formato inválido: {str(e)}")
            return False

    def clear_data(self):
        """Limpa todos os dados"""
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas'])
        self.forecast_results = None
        self.save_data()
        st.success("Dados limpos com sucesso!")

    def run_forecast(self, model_name, horizon, **kwargs):
        """Executa a previsão de demanda"""
        try:
            ts_data = self.data.set_index('Data')['Unidades Vendidas']

            if model_name == "ARIMA":
                order = st.session_state.model_params['ARIMA']['order']
                forecast = self.arima_forecast(ts_data, horizon, order)
            elif model_name == "Holt-Winters":
                params = st.session_state.model_params['Holt-Winters']
                forecast = self.holt_winters_forecast(ts_data, horizon, **params)
            else:
                params = st.session_state.model_params['Random Forest']
                forecast = self.random_forest_forecast(ts_data, horizon, **params)

            self.forecast_results = {
                'model': model_name,
                'horizon': horizon,
                'forecast': forecast,
                'last_date': ts_data.index[-1],
                'execution_date': datetime.now()
            }
            self.save_data()
            st.success("Previsão realizada com sucesso!")
            return True

        except Exception as e:
            st.error(f"Falha na previsão: {str(e)}")
            return False

    def arima_forecast(self, ts_data, horizon, order):
        """Previsão com modelo ARIMA"""
        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        return forecast

    def holt_winters_forecast(self, ts_data, horizon, trend, seasonal, seasonal_periods):
        """Previsão com modelo Holt-Winters"""
        model = ExponentialSmoothing(
            ts_data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(horizon)
        return forecast

    def random_forest_forecast(self, ts_data, horizon, n_estimators, max_depth):
        """Previsão com Random Forest"""
        # Criar features
        df = pd.DataFrame({'y': ts_data})
        for i in range(1, 8):
            df[f'lag_{i}'] = df['y'].shift(i)
        df = df.dropna()

        # Treinar modelo
        X = df.drop('y', axis=1)
        y = df['y']
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X, y)

        # Fazer previsão
        last_values = df.iloc[-1][['y'] + [f'lag_{i}' for i in range(1, 7)]].values
        forecasts = []

        for _ in range(horizon):
            next_pred = model.predict([last_values])[0]
            forecasts.append(next_pred)
            last_values = np.concatenate([[next_pred], last_values[:-1]])

        return pd.Series(
            forecasts,
            index=pd.date_range(
                start=ts_data.index[-1] + timedelta(days=1),
                periods=horizon
            )
        )

    def export_to_excel(self):
        """Exporta dados para Excel"""
        if self.data.empty:
            st.warning("Nenhum dado para exportar")
            return None

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            self.data.to_excel(writer, sheet_name='Dados Históricos', index=False)

            if self.forecast_results:
                forecast_dates = pd.date_range(
                    start=self.forecast_results['last_date'] + timedelta(days=1),
                    periods=self.forecast_results['horizon']
                )
                forecast_df = pd.DataFrame({
                    'Data': forecast_dates,
                    'Unidades Previstas': self.forecast_results['forecast'],
                    'Modelo': self.forecast_results['model']
                })
                forecast_df.to_excel(writer, sheet_name='Previsões', index=False)

        return output

    def export_to_pdf(self):
        """Exporta relatório para PDF"""
        if not self.forecast_results:
            st.warning("Nenhuma previsão disponível para exportar")
            return None

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Cabeçalho
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Relatório de Previsão de Demanda", 0, 1, 'C')
            pdf.ln(10)

            # Informações básicas
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Informações Gerais:", 0, 1)
            pdf.set_font("Arial", size=10)

            pdf.cell(50, 10, "Data do relatório:", 0, 0)
            pdf.cell(0, 10, self.forecast_results['execution_date'].strftime('%d/%m/%Y %H:%M'), 0, 1)

            pdf.cell(50, 10, "Modelo usado:", 0, 0)
            pdf.cell(0, 10, self.forecast_results['model'], 0, 1)

            pdf.cell(50, 10, "Período previsto:", 0, 0)
            pdf.cell(0, 10, f"{self.forecast_results['horizon']} dias", 0, 1)
            pdf.ln(10)

            # Estatísticas
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Estatísticas da Previsão:", 0, 1)
            pdf.set_font("Arial", size=10)

            forecast_data = self.forecast_results['forecast']
            stats = [
                ("Média diária:", f"{forecast_data.mean():.2f} unidades"),
                ("Mínimo diário:", f"{forecast_data.min():.2f} unidades"),
                ("Máximo diário:", f"{forecast_data.max():.2f} unidades"),
                ("Total previsto:", f"{forecast_data.sum():.2f} unidades")
            ]

            for label, value in stats:
                pdf.cell(50, 10, label, 0, 0)
                pdf.cell(0, 10, value, 0, 1)

            pdf.ln(10)

            # Tabela de previsões
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Previsões Diárias:", 0, 1)
            pdf.set_font("Arial", size=10)

            # Cabeçalho da tabela
            pdf.cell(40, 10, "Data", 1, 0, 'C')
            pdf.cell(40, 10, "Unidades Previstas", 1, 1, 'C')

            # Dados da tabela
            forecast_dates = pd.date_range(
                start=self.forecast_results['last_date'] + timedelta(days=1),
                periods=self.forecast_results['horizon']
            )

            for date, value in zip(forecast_dates, forecast_data):
                pdf.cell(40, 10, date.strftime('%d/%m/%Y'), 1, 0, 'C')
                pdf.cell(40, 10, f"{value:.2f}", 1, 1, 'C')

            output = BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            output.write(pdf_bytes)
            return output

        except Exception as e:
            st.error(f"Falha ao gerar PDF: {str(e)}")
            return None


# Criação da interface
def main():
    st.title("🍞 Sistema de Previsão de Demanda - Padaria Master")

    # Inicializa o sistema
    system = DemandForecastSystem()
    system.load_data()

    # Barra lateral
    st.sidebar.header("Menu")
    menu_options = {
        "📊 Dados": show_data_tab,
        "🔮 Previsão": show_forecast_tab,
        "📊 Visualização": show_visualization_tab,
        "📈 Estatísticas": show_stats_tab,
        "📤 Exportar": show_export_tab,
        "⚙ Configurações": show_settings_tab
    }

    selected_tab = st.sidebar.radio("Navegação", list(menu_options.keys()))

    # Exibe a aba selecionada
    menu_options[selected_tab](system)


def show_data_tab(system):
    """Exibe a aba de gerenciamento de dados"""
    st.header("Gerenciamento de Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Importar Dados")
        uploaded_file = st.file_uploader(
            "Carregar arquivo (CSV ou Excel)",
            type=['csv', 'xlsx'],
            help="O arquivo deve conter colunas 'Data' e 'Unidades Vendidas'"
        )

        if uploaded_file:
            if st.button("Importar Arquivo"):
                system.import_data(uploaded_file)

    with col2:
        st.subheader("Inserir Dados Manualmente")
        with st.form("manual_data_form"):
            date = st.date_input("Data", value=datetime.now())
            units = st.number_input("Unidades Vendidas", min_value=0, step=1)

            if st.form_submit_button("Adicionar Dados"):
                if system.add_manual_data(date, units):
                    st.rerun()

    st.divider()
    st.subheader("Visualização dos Dados")

    if not system.data.empty:
        st.dataframe(system.data, use_container_width=True)

        # Mostrar estatísticas básicas
        st.write(f"**Total de registros:** {len(system.data)}")
        st.write(f"**Período:** {system.data['Data'].min().date()} a {system.data['Data'].max().date()}")

        if len(system.data) < 30:
            st.warning("São necessários pelo menos 30 dias de dados para previsões confiáveis.")

        if st.button("Limpar Todos os Dados", type="primary"):
            system.clear_data()
            st.rerun()
    else:
        st.info("Nenhum dado carregado. Importe um arquivo ou insira dados manualmente.")


def show_forecast_tab(system):
    """Exibe a aba de previsão de demanda"""
    st.header("Previsão de Demanda")

    if system.data.empty:
        st.warning("Carregue dados na aba '📊 Dados' antes de executar previsões.")
        return

    if len(system.data) < 30:
        st.warning("Atenção: São recomendados pelo menos 30 dias de dados para previsões confiáveis.")

    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox(
            "Selecione o Modelo",
            ["ARIMA", "Holt-Winters", "Random Forest"],
            help="ARIMA: Modelo estatístico para séries temporais\nHolt-Winters: Modelo com componentes de tendência e sazonalidade\nRandom Forest: Modelo de machine learning"
        )

    with col2:
        horizon = st.selectbox(
            "Horizonte de Previsão (dias)",
            [7, 14, 30],
            help="Número de dias no futuro para prever"
        )

    # Configurações específicas do modelo
    if model == "ARIMA":
        st.subheader("Parâmetros ARIMA")
        col1, col2, col3 = st.columns(3)

        with col1:
            p = st.number_input("Ordem AR (p)", min_value=0, max_value=10, value=5)
        with col2:
            d = st.number_input("Ordem de Diferenciação (d)", min_value=0, max_value=2, value=1)
        with col3:
            q = st.number_input("Ordem MA (q)", min_value=0, max_value=10, value=0)

        st.session_state.model_params['ARIMA']['order'] = (p, d, q)

    elif model == "Holt-Winters":
        st.subheader("Parâmetros Holt-Winters")
        col1, col2 = st.columns(2)

        with col1:
            trend = st.selectbox("Tendência", ['add', 'mul'], help="Tipo de componente de tendência")
            seasonal = st.selectbox("Sazonalidade", ['add', 'mul'], help="Tipo de componente sazonal")
        with col2:
            seasonal_periods = st.number_input(
                "Período Sazonal",
                min_value=2,
                value=7,
                help="Número de períodos em um ciclo sazonal (ex: 7 para semana)"
            )

        st.session_state.model_params['Holt-Winters'] = {
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods
        }

    else:  # Random Forest
        st.subheader("Parâmetros Random Forest")
        col1, col2 = st.columns(2)

        with col1:
            n_estimators = st.number_input(
                "Número de Árvores",
                min_value=10,
                max_value=500,
                value=100
            )
        with col2:
            max_depth = st.number_input(
                "Profundidade Máxima",
                min_value=1,
                max_value=20,
                value=5,
                help="None para profundidade ilimitada"
            )

        st.session_state.model_params['Random Forest'] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }

    if st.button("Executar Previsão", type="primary"):
        with st.spinner(f"Executando previsão com {model}..."):
            if system.run_forecast(model, horizon):
                st.rerun()

    if system.forecast_results:
        st.divider()
        st.subheader("Resultados da Previsão")

        # Gráfico de previsão
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(system.data['Data'], system.data['Unidades Vendidas'], label='Histórico', color='blue')

        forecast_dates = pd.date_range(
            start=system.forecast_results['last_date'] + timedelta(days=1),
            periods=system.forecast_results['horizon']
        )
        ax.plot(forecast_dates, system.forecast_results['forecast'],
                label=f'Previsão ({system.forecast_results["model"]})',
                color='red', linestyle='--')

        ax.set_title(f"Previsão de Demanda - {system.forecast_results['model']}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Unidades Vendidas")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # Tabela de previsões
        forecast_df = pd.DataFrame({
            'Data': forecast_dates,
            'Unidades Previstas': system.forecast_results['forecast']
        })
        st.dataframe(forecast_df, use_container_width=True)


def show_visualization_tab(system):
    """Exibe a aba de visualização de dados"""
    st.header("Visualização de Dados")

    if system.data.empty:
        st.warning("Nenhum dado disponível para visualização")
        return

    # Filtros de data
    min_date = system.data['Data'].min()
    max_date = system.data['Data'].max()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Data Inicial",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )

    with col2:
        end_date = st.date_input(
            "Data Final",
            min_value=min_date,
            max_value=max_date,
            value=max_date
        )

    # Filtrar dados
    filtered_data = system.data[
        (system.data['Data'] >= pd.to_datetime(start_date)) &
        (system.data['Data'] <= pd.to_datetime(end_date))
        ]

    if filtered_data.empty:
        st.warning("Nenhum dado no período selecionado")
        return

    # Gráficos
    tab1, tab2 = st.tabs(["Série Temporal", "Análise por Dia da Semana"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_data['Data'], filtered_data['Unidades Vendidas'])
        ax.set_title("Vendas por Data")
        ax.set_xlabel("Data")
        ax.set_ylabel("Unidades Vendidas")
        ax.grid(True)
        st.pyplot(fig)

    with tab2:
        filtered_data['DiaSemana'] = filtered_data['Data'].dt.day_name()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Boxplot
        sns.boxplot(data=filtered_data, x='DiaSemana', y='Unidades Vendidas', ax=ax1)
        ax1.set_title("Distribuição por Dia da Semana")
        ax1.tick_params(axis='x', rotation=45)

        # Média por dia
        mean_by_day = filtered_data.groupby('DiaSemana')['Unidades Vendidas'].mean()
        mean_by_day.plot(kind='bar', ax=ax2)
        ax2.set_title("Média de Vendas por Dia da Semana")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)


def show_stats_tab(system):
    """Exibe a aba de estatísticas"""
    st.header("Estatísticas da Previsão")

    if not system.forecast_results:
        st.warning("Execute uma previsão na aba '🔮 Previsão' para ver estatísticas")
        return

    forecast_data = system.forecast_results['forecast']

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Média Diária", f"{forecast_data.mean():.1f} unidades")
    col2.metric("Mínimo Diário", f"{forecast_data.min():.1f} unidades")
    col3.metric("Máximo Diário", f"{forecast_data.max():.1f} unidades")
    col4.metric("Total Previsto", f"{forecast_data.sum():.1f} unidades")

    st.divider()

    # Tabela detalhada
    forecast_dates = pd.date_range(
        start=system.forecast_results['last_date'] + timedelta(days=1),
        periods=system.forecast_results['horizon']
    )

    forecast_df = pd.DataFrame({
        'Data': forecast_dates,
        'Unidades Previstas': forecast_data,
        'Dia da Semana': forecast_dates.day_name()
    })

    st.dataframe(
        forecast_df.style.format({
            'Unidades Previstas': '{:.1f}'
        }),
        use_container_width=True,
        hide_index=True
    )


def show_export_tab(system):
    """Exibe a aba de exportação de dados"""
    st.header("Exportar Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Exportar para Excel")
        if st.button("Gerar Arquivo Excel"):
            excel_file = system.export_to_excel()
            if excel_file:
                st.download_button(
                    label="Baixar Excel",
                    data=excel_file,
                    file_name="previsao_demanda.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with col2:
        st.subheader("Exportar para PDF")
        if st.button("Gerar Relatório PDF"):
            pdf_file = system.export_to_pdf()
            if pdf_file:
                st.download_button(
                    label="Baixar PDF",
                    data=pdf_file,
                    file_name="relatorio_previsao.pdf",
                    mime="application/pdf"
                )


def show_settings_tab(system):
    """Exibe a aba de configurações"""
    st.header("Configurações do Sistema")

    st.subheader("Configurações de Modelos")

    with st.expander("Configurações ARIMA"):
        st.write("Parâmetros atuais:", st.session_state.model_params['ARIMA'])

    with st.expander("Configurações Holt-Winters"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Holt-Winters'])

    with st.expander("Configurações Random Forest"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Random Forest'])

    st.subheader("Sobre o Sistema")
    st.write("""
        **Sistema de Previsão de Demanda - Padaria Master**  
        Versão 2.0  
        Desenvolvido para gestão profissional de demanda  
        © 2023 - Todos os direitos reservados
    """)


if __name__ == "__main__":
    main()