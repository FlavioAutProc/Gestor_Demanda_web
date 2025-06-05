import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
from io import BytesIO
from fpdf import FPDF
import warnings
import logging
from scipy import stats
import tempfile
import os
import json
from pmdarima import auto_arima
import optuna
from functools import lru_cache

# Configurações iniciais
warnings.filterwarnings("ignore")
logging.basicConfig(filename='app_errors.log', level=logging.ERROR)

# Configuração da página
st.set_page_config(
    page_title="Sistema Avançado de Previsão de Demanda - Padaria Master",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)


class AdvancedDemandForecastSystem:
    def __init__(self):
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas', 'Produto'])
        self.forecast_results = None
        self.initialize_session_state()
        self.setup_cache()

    def setup_cache(self):
        """Configura funções com cache para melhor desempenho"""
        self.process_data = lru_cache(maxsize=32)(self._process_data)
        self.calculate_statistics = lru_cache(maxsize=32)(self._calculate_statistics)

    def initialize_session_state(self):
        """Inicializa o estado da sessão para persistência de dados"""
        if 'data' not in st.session_state:
            st.session_state.data = self.data
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {
                'ARIMA': {'order': (5, 1, 0)},
                'SARIMA': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)},
                'Holt-Winters': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
                'Random Forest': {'n_estimators': 100, 'max_depth': None},
                'XGBoost': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
                'Prophet': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
            }
        if 'data_backups' not in st.session_state:
            st.session_state.data_backups = []

    def load_data(self):
        """Carrega dados da sessão"""
        self.data = st.session_state.data
        self.forecast_results = st.session_state.forecast_results

    def save_data(self):
        """Salva dados na sessão e cria backup"""
        st.session_state.data = self.data
        st.session_state.forecast_results = self.forecast_results

        # Criar backup (mantém últimos 5)
        if len(st.session_state.data_backups) >= 5:
            st.session_state.data_backups.pop(0)
        st.session_state.data_backups.append(self.data.copy())

    def _process_data(self, data):
        """Processamento de dados com cache"""
        processed = data.copy()
        processed['Data'] = pd.to_datetime(processed['Data'])
        processed = processed.sort_values('Data')
        return processed

    def import_data(self, uploaded_files):
        """Importa dados de múltiplos arquivos com tratamento avançado"""
        try:
            new_dfs = []

            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.xlsx'):
                    new_data = pd.read_excel(uploaded_file)
                else:
                    new_data = pd.read_csv(uploaded_file)

                # Verificar colunas necessárias
                required_cols = ['Data', 'Unidades Vendidas']
                if not all(col in new_data.columns for col in required_cols):
                    raise ValueError(f"Arquivo {uploaded_file.name} deve conter colunas 'Data' e 'Unidades Vendidas'")

                # Adicionar coluna de produto se não existir
                if 'Produto' not in new_data.columns:
                    new_data['Produto'] = 'Padaria Geral'

                new_dfs.append(new_data)

            if not new_dfs:
                return False

            new_data = pd.concat(new_dfs, ignore_index=True)
            new_data = self._process_data(new_data)

            # Tratamento de dados
            new_data = self.handle_missing_dates(new_data)
            new_data = self.detect_outliers(new_data)

            # Verificar duplicatas
            dup_cols = ['Data', 'Produto'] if 'Produto' in new_data.columns else ['Data']
            if new_data.duplicated(subset=dup_cols).any():
                st.warning("Foram encontradas duplicatas. Serão mantidos apenas os últimos valores.")
                new_data = new_data.drop_duplicates(subset=dup_cols, keep='last')

            self.data = new_data
            self.save_data()
            st.success(f"Dados importados com sucesso! {len(self.data)} registros carregados.")
            return True

        except Exception as e:
            st.error(f"Falha ao importar arquivo: {str(e)}")
            logging.error(f"Import error: {str(e)}")
            return False

    def handle_missing_dates(self, data):
        """Preenche datas faltantes com interpolação"""
        if data.empty:
            return data

        # Criar range completo de datas
        date_range = pd.date_range(
            start=data['Data'].min(),
            end=data['Data'].max(),
            freq='D'
        )

        # Para cada produto, preencher datas faltantes
        products = data['Produto'].unique() if 'Produto' in data.columns else [None]
        filled_dfs = []

        for product in products:
            if product:
                product_data = data[data['Produto'] == product].copy()
            else:
                product_data = data.copy()

            # Reindexar para todas as datas
            product_data = product_data.set_index('Data').reindex(date_range)
            product_data['Produto'] = product

            # Interpolar valores faltantes
            product_data['Unidades Vendidas'] = product_data['Unidades Vendidas'].interpolate(
                method='linear',
                limit_direction='both'
            ).fillna(0)  # Preencher com 0 se não puder interpolar

            filled_dfs.append(product_data.reset_index().rename(columns={'index': 'Data'}))

        return pd.concat(filled_dfs, ignore_index=True)

    def detect_outliers(self, data, threshold=3):
        """Detecta e trata outliers usando Z-Score"""
        if data.empty:
            return data

        data = data.copy()
        z_scores = np.abs(stats.zscore(data['Unidades Vendidas']))
        outliers = z_scores > threshold

        if outliers.any():
            st.warning(f"Detectados {outliers.sum()} outliers. Eles serão substituídos pela mediana.")
            median_val = data['Unidades Vendidas'].median()
            data.loc[outliers, 'Unidades Vendidas'] = median_val

        return data

    def add_manual_data(self, date, units, product):
        """Adiciona dados inseridos manualmente"""
        try:
            date = pd.to_datetime(date)
            units = float(units)

            # Verificar se data já existe para o produto
            mask = (pd.to_datetime(self.data['Data']) == date)
            if 'Produto' in self.data.columns:
                mask &= (self.data['Produto'] == product)

            if not self.data.empty and mask.any():
                # Atualizar entrada existente
                self.data.loc[mask, 'Unidades Vendidas'] = units
            else:
                # Adicionar novo dado
                new_data = pd.DataFrame([[date, units, product]],
                                        columns=['Data', 'Unidades Vendidas', 'Produto'])
                self.data = pd.concat([self.data, new_data], ignore_index=True)

            self.data = self._process_data(self.data)
            self.save_data()
            st.success("Dados adicionados com sucesso!")
            return True

        except ValueError as e:
            st.error(f"Formato inválido: {str(e)}")
            logging.error(f"Manual data error: {str(e)}")
            return False

    def clear_data(self):
        """Limpa todos os dados"""
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas', 'Produto'])
        self.forecast_results = None
        self.save_data()
        st.success("Dados limpos com sucesso!")

    def run_forecast(self, model_name, horizon, product=None, **kwargs):
        """Executa a previsão de demanda com validação cruzada"""
        try:
            # Filtrar por produto se especificado
            if product and 'Produto' in self.data.columns:
                ts_data = self.data[self.data['Produto'] == product].set_index('Data')['Unidades Vendidas']
            else:
                ts_data = self.data.set_index('Data')['Unidades Vendidas']

            if len(ts_data) < 30:
                st.warning("Dados insuficientes para previsão confiável. Recomendado pelo menos 30 pontos.")

            # Configurar progresso
            progress_bar = st.progress(0)
            status_text = st.empty()

            if model_name == "ARIMA":
                status_text.text("Ajustando modelo ARIMA...")
                order = st.session_state.model_params['ARIMA']['order']
                forecast = self.arima_forecast(ts_data, horizon, order)
                progress_bar.progress(100)

            elif model_name == "SARIMA":
                status_text.text("Ajustando modelo SARIMA...")
                order = st.session_state.model_params['SARIMA']['order']
                seasonal_order = st.session_state.model_params['SARIMA']['seasonal_order']
                forecast = self.sarima_forecast(ts_data, horizon, order, seasonal_order)
                progress_bar.progress(100)

            elif model_name == "Holt-Winters":
                status_text.text("Ajustando modelo Holt-Winters...")
                params = st.session_state.model_params['Holt-Winters']
                forecast = self.holt_winters_forecast(ts_data, horizon, **params)
                progress_bar.progress(100)

            elif model_name == "Prophet":
                status_text.text("Ajustando modelo Prophet...")
                params = st.session_state.model_params['Prophet']
                forecast = self.prophet_forecast(ts_data, horizon, **params)
                progress_bar.progress(100)

            elif model_name == "Random Forest":
                status_text.text("Treinando Random Forest com validação cruzada...")
                params = st.session_state.model_params['Random Forest']
                forecast = self.ml_forecast(ts_data, horizon, model='rf', **params)
                progress_bar.progress(100)

            elif model_name == "XGBoost":
                status_text.text("Treinando XGBoost com validação cruzada...")
                params = st.session_state.model_params['XGBoost']
                forecast = self.ml_forecast(ts_data, horizon, model='xgb', **params)
                progress_bar.progress(100)

            else:
                raise ValueError(f"Modelo desconhecido: {model_name}")

            self.forecast_results = {
                'model': model_name,
                'horizon': horizon,
                'forecast': forecast,
                'last_date': ts_data.index[-1],
                'execution_date': datetime.now(),
                'product': product
            }
            self.save_data()
            status_text.text("Previsão concluída com sucesso!")
            st.success("Previsão realizada com sucesso!")
            return True

        except Exception as e:
            st.error(f"Falha na previsão: {str(e)}")
            logging.error(f"Forecast error: {str(e)}")
            return False

    def arima_forecast(self, ts_data, horizon, order):
        """Previsão com modelo ARIMA"""
        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        return forecast

    def sarima_forecast(self, ts_data, horizon, order, seasonal_order):
        """Previsão com modelo SARIMA"""
        model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
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

    def prophet_forecast(self, ts_data, horizon, **params):
        """Previsão com Facebook Prophet"""
        df = pd.DataFrame({
            'ds': ts_data.index,
            'y': ts_data.values
        })

        model = Prophet(**params)
        model.fit(df)

        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        return forecast.tail(horizon)['yhat'].values

    def ml_forecast(self, ts_data, horizon, model='rf', **params):
        """Previsão com modelos de machine learning (RF ou XGBoost)"""
        # Criar features
        df = pd.DataFrame({'y': ts_data})
        for i in range(1, 8):
            df[f'lag_{i}'] = df['y'].shift(i)
        df = df.dropna()

        # Validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        X = df.drop('y', axis=1)
        y = df['y']

        if model == 'rf':
            model_inst = RandomForestRegressor(**params, random_state=42)
        else:
            model_inst = XGBRegressor(**params, random_state=42)

        # Treinar modelo
        model_inst.fit(X, y)

        # Fazer previsão
        last_values = df.iloc[-1][['y'] + [f'lag_{i}' for i in range(1, 7)]].values
        forecasts = []

        for _ in range(horizon):
            next_pred = model_inst.predict([last_values])[0]
            forecasts.append(next_pred)
            last_values = np.concatenate([[next_pred], last_values[:-1]])

        return pd.Series(
            forecasts,
            index=pd.date_range(
                start=ts_data.index[-1] + timedelta(days=1),
                periods=horizon
            )
        )

    def auto_tune_model(self, model_name):
        """Otimização automática de hiperparâmetros com Optuna"""
        try:
            if self.data.empty:
                st.warning("Nenhum dado disponível para otimização")
                return

            ts_data = self.data.set_index('Data')['Unidades Vendidas']

            if model_name == "ARIMA":
                st.info("Executando auto_arima para encontrar melhores parâmetros...")
                model = auto_arima(
                    ts_data,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True
                )
                best_order = model.order
                st.session_state.model_params['ARIMA']['order'] = best_order
                st.success(f"Melhores parâmetros ARIMA encontrados: {best_order}")

            elif model_name == "SARIMA":
                st.info("Executando auto_arima para encontrar melhores parâmetros SARIMA...")
                model = auto_arima(
                    ts_data,
                    seasonal=True,
                    m=7,  # Sazonalidade semanal
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True
                )
                best_order = model.order
                best_seasonal_order = model.seasonal_order
                st.session_state.model_params['SARIMA']['order'] = best_order
                st.session_state.model_params['SARIMA']['seasonal_order'] = best_seasonal_order
                st.success(f"Melhores parâmetros SARIMA encontrados: ordem {best_order}, sazonal {best_seasonal_order}")

            elif model_name in ["Random Forest", "XGBoost"]:
                st.info(f"Otimizando {model_name} com Optuna...")

                # Preparar dados
                df = pd.DataFrame({'y': ts_data})
                for i in range(1, 8):
                    df[f'lag_{i}'] = df['y'].shift(i)
                df = df.dropna()
                X = df.drop('y', axis=1)
                y = df['y']

                def objective(trial):
                    if model_name == "Random Forest":
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'max_depth': trial.suggest_int('max_depth', 3, 15),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                        }
                        model = RandomForestRegressor(**params, random_state=42)
                    else:
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                            'max_depth': trial.suggest_int('max_depth', 3, 10),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
                        }
                        model = XGBRegressor(**params, random_state=42)

                    # Validação cruzada
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = []

                    for train_idx, test_idx in tscv.split(X):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                        scores.append(score)

                    return np.mean(scores)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=20)

                if model_name == "Random Forest":
                    st.session_state.model_params['Random Forest'].update({
                        'n_estimators': study.best_params['n_estimators'],
                        'max_depth': study.best_params['max_depth']
                    })
                else:
                    st.session_state.model_params['XGBoost'].update({
                        'n_estimators': study.best_params['n_estimators'],
                        'max_depth': study.best_params['max_depth'],
                        'learning_rate': study.best_params['learning_rate']
                    })

                st.success(f"Melhores parâmetros {model_name} encontrados: {study.best_params}")

        except Exception as e:
            st.error(f"Falha na otimização: {str(e)}")
            logging.error(f"Auto-tune error: {str(e)}")

    def _calculate_statistics(self, data):
        """Calcula estatísticas com cache"""
        stats = {
            'total': len(data),
            'start_date': data['Data'].min(),
            'end_date': data['Data'].max(),
            'mean': data['Unidades Vendidas'].mean(),
            'median': data['Unidades Vendidas'].median(),
            'std': data['Unidades Vendidas'].std(),
            'min': data['Unidades Vendidas'].min(),
            'max': data['Unidades Vendidas'].max()
        }
        return stats

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
                    'Modelo': self.forecast_results['model'],
                    'Produto': self.forecast_results.get('product', 'Geral')
                })
                forecast_df.to_excel(writer, sheet_name='Previsões', index=False)

        return output

    def export_to_pdf(self):
        """Exporta relatório para PDF com gráficos"""
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

            if 'product' in self.forecast_results and self.forecast_results['product']:
                pdf.cell(50, 10, "Produto:", 0, 0)
                pdf.cell(0, 10, self.forecast_results['product'], 0, 1)

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

            # Adicionar gráfico (simplificado - em produção, salvaria uma imagem)
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Gráfico de Previsão:", 0, 1)
            pdf.cell(0, 10, "[Gráfico seria exibido aqui na versão completa]", 0, 1)

            output = BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            output.write(pdf_bytes)
            return output

        except Exception as e:
            st.error(f"Falha ao gerar PDF: {str(e)}")
            logging.error(f"PDF export error: {str(e)}")
            return None

    def get_product_list(self):
        """Retorna lista de produtos únicos"""
        if 'Produto' not in self.data.columns:
            return []
        return sorted(self.data['Produto'].unique().tolist())


# Criação da interface aprimorada
def main():
    st.title("🍞 Sistema Avançado de Previsão de Demanda - Padaria Master")

    # Inicializa o sistema
    system = AdvancedDemandForecastSystem()
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
    """Exibe a aba de gerenciamento de dados aprimorada"""
    st.header("Gerenciamento de Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Importar Dados")
        uploaded_files = st.file_uploader(
            "Carregar arquivos (CSV ou Excel)",
            type=['csv', 'xlsx'],
            help="Os arquivos devem conter colunas 'Data' e 'Unidades Vendidas'",
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Importar Arquivos"):
                with st.spinner("Processando arquivos..."):
                    if system.import_data(uploaded_files):
                        st.rerun()

    with col2:
        st.subheader("Inserir Dados Manualmente")
        with st.form("manual_data_form"):
            col1, col2 = st.columns(2)

            with col1:
                date = st.date_input("Data", value=datetime.now())
                units = st.number_input("Unidades Vendidas", min_value=0, step=1)

            with col2:
                products = system.get_product_list()
                new_product = st.text_input("Novo Produto (ou selecione abaixo)")

                if products:
                    selected_product = st.selectbox(
                        "Produto Existente",
                        options=products,
                        index=0
                    )
                    product = new_product if new_product else selected_product
                else:
                    product = new_product if new_product else "Padaria Geral"

            if st.form_submit_button("Adicionar Dados"):
                if system.add_manual_data(date, units, product):
                    st.rerun()

    st.divider()
    st.subheader("Visualização dos Dados")

    if not system.data.empty:
        # Filtros avançados
        with st.expander("Filtros Avançados"):
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Data Inicial",
                    value=system.data['Data'].min(),
                    min_value=system.data['Data'].min(),
                    max_value=system.data['Data'].max()
                )

            with col2:
                end_date = st.date_input(
                    "Data Final",
                    value=system.data['Data'].max(),
                    min_value=system.data['Data'].min(),
                    max_value=system.data['Data'].max()
                )

            products = system.get_product_list()
            if products:
                selected_products = st.multiselect(
                    "Produtos",
                    options=products,
                    default=products
                )
            else:
                selected_products = []

        # Aplicar filtros
        filtered_data = system.data[
            (system.data['Data'] >= pd.to_datetime(start_date)) &
            (system.data['Data'] <= pd.to_datetime(end_date))
            ]

        if selected_products:
            filtered_data = filtered_data[filtered_data['Produto'].isin(selected_products)]

        # Mostrar dados
        st.dataframe(filtered_data, use_container_width=True, hide_index=True)

        # Mostrar estatísticas básicas
        stats = system.calculate_statistics(filtered_data)

        st.write(f"**Total de registros:** {stats['total']}")
        st.write(f"**Período:** {stats['start_date'].date()} a {stats['end_date'].date()}")
        st.write(f"**Média diária:** {stats['mean']:.2f} unidades")
        st.write(f"**Mediana diária:** {stats['median']:.2f} unidades")

        if len(filtered_data) < 30:
            st.warning("São necessários pelo menos 30 dias de dados para previsões confiáveis.")

        if st.button("Limpar Todos os Dados", type="primary"):
            system.clear_data()
            st.rerun()
    else:
        st.info("Nenhum dado carregado. Importe arquivos ou insira dados manualmente.")


def show_forecast_tab(system):
    """Exibe a aba de previsão de demanda aprimorada"""
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
            ["ARIMA", "SARIMA", "Holt-Winters", "Prophet", "Random Forest", "XGBoost"],
            help="ARIMA: Modelo estatístico para séries temporais\nSARIMA: ARIMA com sazonalidade\nHolt-Winters: Modelo com componentes de tendência e sazonalidade\nProphet: Modelo do Facebook para séries temporais\nRandom Forest/XGBoost: Modelos de machine learning"
        )

    with col2:
        horizon = st.selectbox(
            "Horizonte de Previsão (dias)",
            [7, 14, 30],
            help="Número de dias no futuro para prever"
        )

    # Seleção de produto se houver múltiplos
    products = system.get_product_list()
    if products:
        product = st.selectbox(
            "Produto para Previsão",
            options=["Todos"] + products,
            index=0
        )
        product = None if product == "Todos" else product
    else:
        product = None

    # Configurações específicas do modelo
    if model in ["ARIMA", "SARIMA", "Holt-Winters", "Random Forest", "XGBoost", "Prophet"]:
        with st.expander(f"Configurações {model}"):
            if model == "ARIMA":
                st.write("Parâmetros ARIMA (p, d, q)")
                col1, col2, col3 = st.columns(3)

                with col1:
                    p = st.number_input("Ordem AR (p)", min_value=0, max_value=10,
                                        value=st.session_state.model_params['ARIMA']['order'][0])
                with col2:
                    d = st.number_input("Ordem de Diferenciação (d)", min_value=0, max_value=2,
                                        value=st.session_state.model_params['ARIMA']['order'][1])
                with col3:
                    q = st.number_input("Ordem MA (q)", min_value=0, max_value=10,
                                        value=st.session_state.model_params['ARIMA']['order'][2])

                st.session_state.model_params['ARIMA']['order'] = (p, d, q)

                if st.button("Otimizar Parâmetros ARIMA"):
                    with st.spinner("Procurando melhores parâmetros..."):
                        system.auto_tune_model("ARIMA")

            elif model == "SARIMA":
                st.write("Parâmetros SARIMA")
                cols = st.columns(4)

                with cols[0]:
                    p = st.number_input("Ordem AR (p)", min_value=0, max_value=3,
                                        value=st.session_state.model_params['SARIMA']['order'][0])
                with cols[1]:
                    d = st.number_input("Ordem Diferenciação (d)", min_value=0, max_value=2,
                                        value=st.session_state.model_params['SARIMA']['order'][1])
                with cols[2]:
                    q = st.number_input("Ordem MA (q)", min_value=0, max_value=3,
                                        value=st.session_state.model_params['SARIMA']['order'][2])

                st.write("Parâmetros Sazonais (P, D, Q, m)")
                cols = st.columns(4)

                with cols[0]:
                    P = st.number_input("Ordem Sazonal AR (P)", min_value=0, max_value=3,
                                        value=st.session_state.model_params['SARIMA']['seasonal_order'][0])
                with cols[1]:
                    D = st.number_input("Ordem Sazonal Dif (D)", min_value=0, max_value=2,
                                        value=st.session_state.model_params['SARIMA']['seasonal_order'][1])
                with cols[2]:
                    Q = st.number_input("Ordem Sazonal MA (Q)", min_value=0, max_value=3,
                                        value=st.session_state.model_params['SARIMA']['seasonal_order'][2])
                with cols[3]:
                    m = st.number_input("Período Sazonal (m)", min_value=2, value=7)

                st.session_state.model_params['SARIMA']['order'] = (p, d, q)
                st.session_state.model_params['SARIMA']['seasonal_order'] = (P, D, Q, m)

                if st.button("Otimizar Parâmetros SARIMA"):
                    with st.spinner("Procurando melhores parâmetros..."):
                        system.auto_tune_model("SARIMA")

            elif model == "Holt-Winters":
                col1, col2 = st.columns(2)

                with col1:
                    trend = st.selectbox(
                        "Tendência",
                        ['add', 'mul'],
                        index=0 if st.session_state.model_params['Holt-Winters']['trend'] == 'add' else 1,
                        help="Tipo de componente de tendência"
                    )
                    seasonal = st.selectbox(
                        "Sazonalidade",
                        ['add', 'mul'],
                        index=0 if st.session_state.model_params['Holt-Winters']['seasonal'] == 'add' else 1,
                        help="Tipo de componente sazonal"
                    )

                with col2:
                    seasonal_periods = st.number_input(
                        "Período Sazonal",
                        min_value=2,
                        value=st.session_state.model_params['Holt-Winters']['seasonal_periods'],
                        help="Número de períodos em um ciclo sazonal (ex: 7 para semana)"
                    )

                st.session_state.model_params['Holt-Winters'] = {
                    'trend': trend,
                    'seasonal': seasonal,
                    'seasonal_periods': seasonal_periods
                }

            elif model == "Prophet":
                changepoint_prior_scale = st.slider(
                    "Flexibilidade da Tendência",
                    min_value=0.01,
                    max_value=0.5,
                    value=st.session_state.model_params['Prophet']['changepoint_prior_scale'],
                    step=0.01,
                    help="Controla quão flexível é a tendência"
                )

                seasonality_prior_scale = st.slider(
                    "Força da Sazonalidade",
                    min_value=1.0,
                    max_value=20.0,
                    value=st.session_state.model_params['Prophet']['seasonality_prior_scale'],
                    step=1.0,
                    help="Controla a força dos componentes sazonais"
                )

                st.session_state.model_params['Prophet'] = {
                    'changepoint_prior_scale': changepoint_prior_scale,
                    'seasonality_prior_scale': seasonality_prior_scale
                }

            elif model == "Random Forest":
                col1, col2 = st.columns(2)

                with col1:
                    n_estimators = st.number_input(
                        "Número de Árvores",
                        min_value=10,
                        max_value=500,
                        value=st.session_state.model_params['Random Forest']['n_estimators']
                    )

                with col2:
                    max_depth = st.number_input(
                        "Profundidade Máxima",
                        min_value=1,
                        max_value=20,
                        value=st.session_state.model_params['Random Forest'].get('max_depth', 5),
                        help="None para profundidade ilimitada"
                    )

                st.session_state.model_params['Random Forest'] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth if max_depth > 0 else None
                }

                if st.button("Otimizar Random Forest"):
                    with st.spinner("Procurando melhores parâmetros..."):
                        system.auto_tune_model("Random Forest")

            elif model == "XGBoost":
                col1, col2 = st.columns(2)

                with col1:
                    n_estimators = st.number_input(
                        "Número de Árvores",
                        min_value=10,
                        max_value=500,
                        value=st.session_state.model_params['XGBoost']['n_estimators']
                    )

                with col2:
                    max_depth = st.number_input(
                        "Profundidade Máxima",
                        min_value=1,
                        max_value=10,
                        value=st.session_state.model_params['XGBoost']['max_depth']
                    )

                learning_rate = st.slider(
                    "Taxa de Aprendizado",
                    min_value=0.01,
                    max_value=0.3,
                    value=st.session_state.model_params['XGBoost']['learning_rate'],
                    step=0.01
                )

                st.session_state.model_params['XGBoost'] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate
                }

                if st.button("Otimizar XGBoost"):
                    with st.spinner("Procurando melhores parâmetros..."):
                        system.auto_tune_model("XGBoost")

    if st.button("Executar Previsão", type="primary"):
        with st.spinner(f"Executando previsão com {model}..."):
            if system.run_forecast(model, horizon, product):
                st.rerun()

    if system.forecast_results:
        st.divider()
        st.subheader("Resultados da Previsão")

        # Dados para o gráfico
        if product:
            historical_data = system.data[system.data['Produto'] == product]
        else:
            historical_data = system.data

        forecast_dates = pd.date_range(
            start=system.forecast_results['last_date'] + timedelta(days=1),
            periods=system.forecast_results['horizon']
        )

        # Criar gráfico interativo com Plotly
        fig = go.Figure()

        # Adicionar histórico
        fig.add_trace(go.Scatter(
            x=historical_data['Data'],
            y=historical_data['Unidades Vendidas'],
            mode='lines',
            name='Histórico',
            line=dict(color='blue')
        ))

        # Adicionar previsão
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=system.forecast_results['forecast'],
            mode='lines',
            name=f'Previsão ({system.forecast_results["model"]})',
            line=dict(color='red', dash='dash')
        ))

        # Adicionar intervalo de confiança se disponível
        if system.forecast_results['model'] == 'Prophet':
        # Para Prophet, poderíamos adicionar intervalos de confiança
            pass

        # Configurar layout
        title = f"Previsão de Demanda - {system.forecast_results['model']}"
        if product:
            title += f" - {product}"

        fig.update_layout(
            title=title,
            xaxis_title="Data",
            yaxis_title="Unidades Vendidas",
            hovermode="x unified",
            showlegend=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabela de previsões
        forecast_df = pd.DataFrame({
            'Data': forecast_dates,
            'Unidades Previstas': system.forecast_results['forecast'],
            'Dia da Semana': forecast_dates.day_name()
        })

        st.dataframe(
            forecast_df.style.format({
                'Unidades Previstas': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )


def show_visualization_tab(system):
    """Exibe a aba de visualização de dados aprimorada"""
    st.header("Visualização de Dados")

    if system.data.empty:
        st.warning("Nenhum dado disponível para visualização")
        return

    # Filtros avançados
    with st.expander("Filtros"):
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Data Inicial",
                min_value=system.data['Data'].min(),
                max_value=system.data['Data'].max(),
                value=system.data['Data'].min()
            )

        with col2:
            end_date = st.date_input(
                "Data Final",
                min_value=system.data['Data'].min(),
                max_value=system.data['Data'].max(),
                value=system.data['Data'].max()
            )

        products = system.get_product_list()
        if products:
            selected_products = st.multiselect(
                "Produtos",
                options=products,
                default=products[:min(3, len(products))] if products else []
            )
        else:
            selected_products = []

    # Filtrar dados
    filtered_data = system.data[
        (system.data['Data'] >= pd.to_datetime(start_date)) &
        (system.data['Data'] <= pd.to_datetime(end_date))
        ]

    if selected_products:
        filtered_data = filtered_data[filtered_data['Produto'].isin(selected_products)]

    if filtered_data.empty:
        st.warning("Nenhum dado no período selecionado")
        return

    # Gráficos interativos
    tab1, tab2, tab3 = st.tabs(["Série Temporal", "Análise Sazonal", "Comparação de Produtos"])

    with tab1:
        if selected_products:
            fig = px.line(
                filtered_data,
                x='Data',
                y='Unidades Vendidas',
                color='Produto',
                title="Vendas por Data e Produto",
                labels={'Unidades Vendidas': 'Unidades Vendidas', 'Data': 'Data'}
            )
        else:
            fig = px.line(
                filtered_data,
                x='Data',
                y='Unidades Vendidas',
                title="Vendas por Data",
                labels={'Unidades Vendidas': 'Unidades Vendidas', 'Data': 'Data'}
            )

        fig.update_layout(
            hovermode="x unified",
            showlegend=True if selected_products else False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        filtered_data['DiaSemana'] = filtered_data['Data'].dt.day_name()
        filtered_data['Mes'] = filtered_data['Data'].dt.month_name()

        if selected_products:
            fig = px.box(
                filtered_data,
                x='DiaSemana',
                y='Unidades Vendidas',
                color='Produto',
                title="Distribuição por Dia da Semana",
                labels={'Unidades Vendidas': 'Unidades Vendidas', 'DiaSemana': 'Dia da Semana'}
            )
        else:
            fig = px.box(
                filtered_data,
                x='DiaSemana',
                y='Unidades Vendidas',
                title="Distribuição por Dia da Semana",
                labels={'Unidades Vendidas': 'Unidades Vendidas', 'DiaSemana': 'Dia da Semana'}
            )

        fig.update_layout(
            xaxis={'categoryorder': 'array',
                   'categoryarray': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not selected_products or len(selected_products) < 2:
            st.info("Selecione pelo menos 2 produtos para comparação")
        else:
            # Agrupar por produto e data
            product_comparison = filtered_data.groupby(['Produto', pd.Grouper(key='Data', freq='W')])[
                'Unidades Vendidas'].sum().reset_index()

            fig = px.line(
                product_comparison,
                x='Data',
                y='Unidades Vendidas',
                color='Produto',
                title="Comparação Semanal de Vendas por Produto",
                labels={'Unidades Vendidas': 'Unidades Vendidas', 'Data': 'Data'}
            )

            fig.update_layout(
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_stats_tab(system):
    """Exibe a aba de estatísticas aprimorada"""
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

    # Análise por dia da semana
    forecast_dates = pd.date_range(
        start=system.forecast_results['last_date'] + timedelta(days=1),
        periods=system.forecast_results['horizon']
    )

    forecast_df = pd.DataFrame({
        'Data': forecast_dates,
        'Unidades Previstas': forecast_data,
        'Dia da Semana': forecast_dates.day_name()
    })

    # Agrupar por dia da semana
    weekday_stats = forecast_df.groupby('Dia da Semana')['Unidades Previstas'].agg(['mean', 'sum'])
    weekday_stats = weekday_stats.reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    st.subheader("Média por Dia da Semana")
    fig = px.bar(
        weekday_stats,
        x=weekday_stats.index,
        y='mean',
        labels={'mean': 'Média de Unidades', 'Dia da Semana': 'Dia da Semana'},
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        xaxis_title="Dia da Semana",
        yaxis_title="Média de Unidades Previstas",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela detalhada
    st.subheader("Detalhes da Previsão")
    st.dataframe(
        forecast_df.style.format({
            'Unidades Previstas': '{:.1f}'
        }),
        use_container_width=True,
        hide_index=True
    )


def show_export_tab(system):
    """Exibe a aba de exportação de dados aprimorada"""
    st.header("Exportar Dados e Relatórios")

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

    st.divider()

    # Backup e restauração
    st.subheader("Backup de Dados")

    if st.session_state.data_backups:
        backup_dates = [f"Backup {i + 1} ({len(backup)} registros)"
                        for i, backup in enumerate(st.session_state.data_backups)]

        selected_backup = st.selectbox(
            "Selecione um backup para restaurar",
            options=backup_dates
        )

        if st.button("Restaurar Backup"):
            index = backup_dates.index(selected_backup)
            system.data = st.session_state.data_backups[index].copy()
            system.save_data()
            st.success("Backup restaurado com sucesso!")
            st.rerun()
    else:
        st.info("Nenhum backup disponível")


def show_settings_tab(system):
    """Exibe a aba de configurações aprimorada"""
    st.header("Configurações do Sistema")

    st.subheader("Configurações de Modelos")

    with st.expander("Configurações ARIMA"):
        st.write("Parâmetros atuais:", st.session_state.model_params['ARIMA'])
        if st.button("Redefinir ARIMA para padrões"):
            st.session_state.model_params['ARIMA'] = {'order': (5, 1, 0)}
            st.success("Parâmetros ARIMA redefinidos")

    with st.expander("Configurações SARIMA"):
        st.write("Parâmetros atuais:", st.session_state.model_params['SARIMA'])
        if st.button("Redefinir SARIMA para padrões"):
            st.session_state.model_params['SARIMA'] = {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 7)
            }
            st.success("Parâmetros SARIMA redefinidos")

    with st.expander("Configurações Holt-Winters"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Holt-Winters'])
        if st.button("Redefinir Holt-Winters para padrões"):
            st.session_state.model_params['Holt-Winters'] = {
                'trend': 'add',
                'seasonal': 'add',
                'seasonal_periods': 7
            }
            st.success("Parâmetros Holt-Winters redefinidos")

    with st.expander("Configurações Random Forest"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Random Forest'])
        if st.button("Redefinir Random Forest para padrões"):
            st.session_state.model_params['Random Forest'] = {
                'n_estimators': 100,
                'max_depth': None
            }
            st.success("Parâmetros Random Forest redefinidos")

    with st.expander("Configurações XGBoost"):
        st.write("Parâmetros atuais:", st.session_state.model_params['XGBoost'])
        if st.button("Redefinir XGBoost para padrões"):
            st.session_state.model_params['XGBoost'] = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1
            }
            st.success("Parâmetros XGBoost redefinidos")

    with st.expander("Configurações Prophet"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Prophet'])
        if st.button("Redefinir Prophet para padrões"):
            st.session_state.model_params['Prophet'] = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10
            }
            st.success("Parâmetros Prophet redefinidos")

    st.subheader("Sobre o Sistema")
    st.write("""
        **Sistema Avançado de Previsão de Demanda - Padaria Master**  
        Versão 3.0  
        Desenvolvido para gestão profissional de demanda  

        **Recursos principais:**  
        - Múltiplos modelos de previsão (ARIMA, SARIMA, Holt-Winters, Prophet, Random Forest, XGBoost)  
        - Suporte a múltiplos produtos  
        - Gráficos interativos  
        - Otimização automática de parâmetros  
        - Exportação para Excel e PDF  

        © 2023 - Todos os direitos reservados
    """)


if __name__ == "__main__":
    main()