import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from io import BytesIO
from fpdf import FPDF
import warnings
import logging
import tempfile
import os
from scipy import stats
import optuna
from pmdarima import auto_arima
import joblib

# Configurações iniciais
warnings.filterwarnings("ignore")
logging.basicConfig(filename='demand_forecast.log', level=logging.ERROR)

# Configuração da página
st.set_page_config(
    page_title="Sistema Avançado de Previsão de Demanda ",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)


class AdvancedDemandForecastSystem:
    def __init__(self):
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas'])
        self.forecast_results = None
        self.model_performance = {}
        self.initialize_session_state()
        self.setup_data_validation()

    def initialize_session_state(self):
        """Inicializa o estado da sessão para persistência de dados"""
        if 'data' not in st.session_state:
            st.session_state.data = self.data
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = {}
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {
                'ARIMA': {'order': (5, 1, 0)},
                'SARIMA': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)},
                'Holt-Winters': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
                'Random Forest': {'n_estimators': 100, 'max_depth': None},
                'XGBoost': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
                'Prophet': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
            }
        if 'data_quality_report' not in st.session_state:
            st.session_state.data_quality_report = None

    def setup_data_validation(self):
        """Configura regras de validação de dados"""
        self.validation_rules = {
            'negative_values': lambda df: (df['Unidades Vendidas'] < 0).sum(),
            'missing_dates': self._check_missing_dates,
            'outliers': self._detect_outliers
        }

    def _check_missing_dates(self, df):
        """Verifica datas faltantes na série temporal"""
        if len(df) < 2:
            return 0
        full_date_range = pd.date_range(start=df['Data'].min(), end=df['Data'].max())
        missing_dates = full_date_range.difference(df['Data'])
        return len(missing_dates)

    def _detect_outliers(self, df):
        """Detecta outliers usando IQR"""
        if len(df) < 10:
            return 0
        q1 = df['Unidades Vendidas'].quantile(0.25)
        q3 = df['Unidades Vendidas'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return ((df['Unidades Vendidas'] < lower_bound) | (df['Unidades Vendidas'] > upper_bound)).sum()

    def load_data(self):
        """Carrega dados da sessão"""
        self.data = st.session_state.data
        self.forecast_results = st.session_state.forecast_results
        self.model_performance = st.session_state.model_performance
        self.data_quality_report = st.session_state.data_quality_report

    def save_data(self):
        """Salva dados na sessão"""
        st.session_state.data = self.data
        st.session_state.forecast_results = self.forecast_results
        st.session_state.model_performance = self.model_performance
        st.session_state.data_quality_report = self.data_quality_report

    def validate_data(self, df):
        """Realiza validação de qualidade dos dados"""
        report = {}
        for name, func in self.validation_rules.items():
            report[name] = func(df)
        return report

    def preprocess_data(self, df):
        """Preprocessa os dados antes da análise"""
        # Preenche datas faltantes
        if len(df) > 1:
            full_date_range = pd.date_range(start=df['Data'].min(), end=df['Data'].max(), name='Data')
            df = df.set_index('Data').reindex(full_date_range).reset_index()
            df['Unidades Vendidas'] = df['Unidades Vendidas'].interpolate(method='linear')

        # Remove outliers usando Z-score (para séries com mais de 30 pontos)
        if len(df) > 30:
            z_scores = np.abs(stats.zscore(df['Unidades Vendidas'].dropna()))
            df.loc[z_scores > 3, 'Unidades Vendidas'] = np.nan
            df['Unidades Vendidas'] = df['Unidades Vendidas'].interpolate(method='linear')

        return df.dropna().reset_index(drop=True)

    def import_data(self, uploaded_files):
        """Importa dados de um ou mais arquivos"""
        try:
            dfs = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)

                # Verificar colunas necessárias
                if 'Data' not in df.columns or 'Unidades Vendidas' not in df.columns:
                    st.error(f"O arquivo {uploaded_file.name} deve conter colunas 'Data' e 'Unidades Vendidas'")
                    continue

                # Converter data para datetime
                df['Data'] = pd.to_datetime(df['Data'])
                dfs.append(df)

            if not dfs:
                return False

            new_data = pd.concat(dfs, ignore_index=True)

            # Ordenar por data
            new_data = new_data.sort_values('Data')

            # Verificar duplicatas
            if new_data['Data'].duplicated().any():
                st.warning("Foram encontradas datas duplicadas. Serão mantidos apenas os últimos valores.")
                new_data = new_data.drop_duplicates('Data', keep='last')

            # Pré-processamento
            new_data = self.preprocess_data(new_data)

            # Validação de dados
            self.data_quality_report = self.validate_data(new_data)

            self.data = new_data
            self.save_data()

            # Backup automático
            self.create_backup()

            st.success(f"Dados importados com sucesso! {len(self.data)} registros carregados.")
            return True

        except Exception as e:
            st.error(f"Falha ao importar arquivo: {str(e)}")
            logging.error(f"Erro na importação de dados: {str(e)}")
            return False

    def create_backup(self):
        """Cria backup dos dados"""
        try:
            backup_dir = os.path.join(tempfile.gettempdir(), "demand_forecast_backups")
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"backup_{timestamp}.csv")

            self.data.to_csv(backup_path, index=False)
        except Exception as e:
            logging.error(f"Erro ao criar backup: {str(e)}")

    def add_manual_data(self, date, units):
        """Adiciona dados inseridos manualmente"""
        try:
            date = pd.to_datetime(date)
            units = float(units)

            if units < 0:
                st.error("Valores negativos não são permitidos")
                return False

            # Verificar se data já existe
            if not self.data.empty and date in pd.to_datetime(self.data['Data']).values:
                # Remover entrada existente
                self.data = self.data[pd.to_datetime(self.data['Data']) != date]

            # Adicionar novo dado
            new_data = pd.DataFrame([[date, units]], columns=['Data', 'Unidades Vendidas'])
            self.data = pd.concat([self.data, new_data], ignore_index=True)
            self.data.sort_values('Data', inplace=True)

            # Pré-processamento após adição
            self.data = self.preprocess_data(self.data)

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
        self.model_performance = {}
        self.data_quality_report = None
        self.save_data()
        st.success("Dados limpos com sucesso!")

    def run_forecast(self, model_name, horizon, **kwargs):
        """Executa a previsão de demanda"""
        try:
            ts_data = self.data.set_index('Data')['Unidades Vendidas']

            if len(ts_data) < 30:
                st.warning("Série temporal muito curta para previsões confiáveis")
                return False

            with st.spinner(f"Treinando modelo {model_name}..."):
                progress_bar = st.progress(0)

                if model_name == "ARIMA":
                    order = st.session_state.model_params['ARIMA']['order']
                    forecast, performance = self.arima_forecast(ts_data, horizon, order)
                elif model_name == "SARIMA":
                    order = st.session_state.model_params['SARIMA']['order']
                    seasonal_order = st.session_state.model_params['SARIMA']['seasonal_order']
                    forecast, performance = self.sarima_forecast(ts_data, horizon, order, seasonal_order)
                elif model_name == "Holt-Winters":
                    params = st.session_state.model_params['Holt-Winters']
                    forecast, performance = self.holt_winters_forecast(ts_data, horizon, **params)
                elif model_name == "Random Forest":
                    params = st.session_state.model_params['Random Forest']
                    forecast, performance = self.random_forest_forecast(ts_data, horizon, **params)
                elif model_name == "XGBoost":
                    params = st.session_state.model_params['XGBoost']
                    forecast, performance = self.xgboost_forecast(ts_data, horizon, **params)
                elif model_name == "Prophet":
                    params = st.session_state.model_params['Prophet']
                    forecast, performance = self.prophet_forecast(ts_data, horizon, **params)
                else:
                    st.error("Modelo não reconhecido")
                    return False

                progress_bar.progress(100)

            self.forecast_results = {
                'model': model_name,
                'horizon': horizon,
                'forecast': forecast,
                'last_date': ts_data.index[-1],
                'execution_date': datetime.now(),
                'performance': performance
            }

            # Atualiza histórico de performance
            if model_name not in self.model_performance:
                self.model_performance[model_name] = []
            self.model_performance[model_name].append({
                'date': datetime.now(),
                'horizon': horizon,
                'mae': performance['mae'],
                'rmse': performance['rmse']
            })

            self.save_data()
            st.success("Previsão realizada com sucesso!")
            return True

        except Exception as e:
            st.error(f"Falha na previsão: {str(e)}")
            logging.error(f"Erro na previsão com {model_name}: {str(e)}")
            return False

    def evaluate_model(self, y_true, y_pred):
        """Avalia o desempenho do modelo"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return {'mae': mae, 'rmse': rmse}

    def cross_validate(self, ts_data, model_func, horizon=7, n_splits=3):
        """Realiza validação cruzada temporal"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []

        for train_index, test_index in tscv.split(ts_data):
            train = ts_data.iloc[train_index]
            test = ts_data.iloc[test_index]

            # Garante que estamos prevendo o mesmo horizonte
            if len(test) > horizon:
                test = test.iloc[:horizon]
            elif len(test) < horizon:
                continue

            forecast = model_func(train, horizon)
            metrics.append(self.evaluate_model(test, forecast))

        if not metrics:
            return {'mae': np.nan, 'rmse': np.nan}

        return {
            'mae': np.mean([m['mae'] for m in metrics]),
            'rmse': np.mean([m['rmse'] for m in metrics])
        }

    def arima_forecast(self, ts_data, horizon, order):
        """Previsão com modelo ARIMA"""

        def model_func(train, h):
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            return model_fit.forecast(steps=h)

        # Cross-validation
        cv_metrics = self.cross_validate(ts_data, model_func, horizon)

        # Fit final
        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)

        return forecast, cv_metrics

    def sarima_forecast(self, ts_data, horizon, order, seasonal_order):
        """Previsão com modelo SARIMA"""

        def model_func(train, h):
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            return model_fit.forecast(steps=h)

        # Cross-validation
        cv_metrics = self.cross_validate(ts_data, model_func, horizon)

        # Fit final
        model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=horizon)

        return forecast, cv_metrics

    def holt_winters_forecast(self, ts_data, horizon, trend, seasonal, seasonal_periods):
        """Previsão com modelo Holt-Winters"""

        def model_func(train, h):
            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            model_fit = model.fit()
            return model_fit.forecast(h)

        # Cross-validation
        cv_metrics = self.cross_validate(ts_data, model_func, horizon)

        # Fit final
        model = ExponentialSmoothing(
            ts_data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(horizon)

        return forecast, cv_metrics

    def random_forest_forecast(self, ts_data, horizon, n_estimators, max_depth):
        """Previsão com Random Forest"""

        def model_func(train, h):
            # Criar features
            df = pd.DataFrame({'y': train})
            for i in range(1, 8):
                df[f'lag_{i}'] = df['y'].shift(i)
            df = df.dropna()

            # Treinar modelo
            X = df.drop('y', axis=1)
            y = df['y']
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1  # Usa todos os cores disponíveis
            )
            model.fit(X, y)

            # Fazer previsão
            last_values = df.iloc[-1][['y'] + [f'lag_{i}' for i in range(1, 7)]].values
            forecasts = []

            for _ in range(h):
                next_pred = model.predict([last_values])[0]
                forecasts.append(next_pred)
                last_values = np.concatenate([[next_pred], last_values[:-1]])

            return pd.Series(forecasts)

        # Cross-validation
        cv_metrics = self.cross_validate(ts_data, model_func, horizon)

        # Fit final
        forecast = model_func(ts_data, horizon)

        return forecast, cv_metrics

    def xgboost_forecast(self, ts_data, horizon, n_estimators, max_depth, learning_rate):
        """Previsão com XGBoost"""

        def model_func(train, h):
            # Criar features
            df = pd.DataFrame({'y': train})
            for i in range(1, 8):
                df[f'lag_{i}'] = df['y'].shift(i)
            df = df.dropna()

            # Treinar modelo
            X = df.drop('y', axis=1)
            y = df['y']
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)

            # Fazer previsão
            last_values = df.iloc[-1][['y'] + [f'lag_{i}' for i in range(1, 7)]].values
            forecasts = []

            for _ in range(h):
                next_pred = model.predict([last_values])[0]
                forecasts.append(next_pred)
                last_values = np.concatenate([[next_pred], last_values[:-1]])

            return pd.Series(forecasts)

        # Cross-validation
        cv_metrics = self.cross_validate(ts_data, model_func, horizon)

        # Fit final
        forecast = model_func(ts_data, horizon)

        return forecast, cv_metrics

    def prophet_forecast(self, ts_data, horizon, changepoint_prior_scale, seasonality_prior_scale):
        """Previsão com Facebook Prophet"""

        def model_func(train, h):
            df = train.reset_index()
            df.columns = ['ds', 'y']

            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=h)
            forecast = model.predict(future)

            return forecast.tail(h)['yhat'].values

        # Cross-validation
        cv_metrics = self.cross_validate(ts_data, model_func, horizon)

        # Fit final
        forecast_values = model_func(ts_data, horizon)
        forecast_dates = pd.date_range(
            start=ts_data.index[-1] + timedelta(days=1),
            periods=horizon
        )
        forecast = pd.Series(forecast_values, index=forecast_dates)

        return forecast, cv_metrics

    def auto_tune_arima(self):
        """Ajuste automático de hiperparâmetros ARIMA"""
        if self.data.empty or len(self.data) < 30:
            st.warning("São necessários pelo menos 30 dias de dados para o ajuste automático")
            return

        with st.spinner("Executando ajuste automático de ARIMA..."):
            ts_data = self.data.set_index('Data')['Unidades Vendidas']

            # Usando pmdarima para encontrar os melhores parâmetros
            model = auto_arima(
                ts_data,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=1, max_d=2,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            st.session_state.model_params['ARIMA']['order'] = model.order
            st.success(f"Melhores parâmetros encontrados: ARIMA{model.order}")

    def auto_tune_sarima(self):
        """Ajuste automático de hiperparâmetros SARIMA"""
        if self.data.empty or len(self.data) < 90:
            st.warning("São necessários pelo menos 90 dias de dados para o ajuste automático SARIMA")
            return

        with st.spinner("Executando ajuste automático de SARIMA..."):
            ts_data = self.data.set_index('Data')['Unidades Vendidas']

            # Usando pmdarima para encontrar os melhores parâmetros
            model = auto_arima(
                ts_data,
                start_p=1, start_q=1,
                max_p=2, max_q=2,
                d=1, max_d=2,
                start_P=1, start_Q=1,
                max_P=2, max_Q=2,
                D=1, max_D=2,
                m=7,  # Sazonalidade semanal
                seasonal=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            st.session_state.model_params['SARIMA']['order'] = model.order
            st.session_state.model_params['SARIMA']['seasonal_order'] = model.seasonal_order
            st.success(f"Melhores parâmetros encontrados: SARIMA{model.order}{model.seasonal_order}")

    def auto_tune_prophet(self):
        """Ajuste automático de hiperparâmetros do Prophet"""
        if self.data.empty or len(self.data) < 60:
            st.warning("São necessários pelo menos 60 dias de dados para o ajuste automático do Prophet")
            return

        with st.spinner("Executando ajuste automático do Prophet..."):
            ts_data = self.data.set_index('Data')['Unidades Vendidas']
            df = ts_data.reset_index()
            df.columns = ['ds', 'y']

            def objective(trial):
                changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5)
                seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 50)

                # Validação cruzada
                tscv = TimeSeriesSplit(n_splits=3)
                mae_scores = []

                for train_index, test_index in tscv.split(ts_data):
                    train = ts_data.iloc[train_index].reset_index()
                    train.columns = ['ds', 'y']
                    test = ts_data.iloc[test_index]

                    if len(test) < 7:
                        continue

                    test = test.iloc[:7]  # Avaliamos apenas 7 dias para consistência

                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=False
                    )
                    model.fit(train)

                    future = model.make_future_dataframe(periods=7)
                    forecast = model.predict(future)

                    pred = forecast.tail(7)['yhat'].values
                    mae = mean_absolute_error(test, pred)
                    mae_scores.append(mae)

                if not mae_scores:
                    return float('inf')

                return np.mean(mae_scores)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)

            best_params = study.best_params
            st.session_state.model_params['Prophet'] = best_params
            st.success(f"Melhores parâmetros encontrados: {best_params}")

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
                    'MAE': self.forecast_results['performance']['mae'],
                    'RMSE': self.forecast_results['performance']['rmse']
                })
                forecast_df.to_excel(writer, sheet_name='Previsões', index=False)

            # Adicionar métricas de performance
            if self.model_performance:
                for model_name, runs in self.model_performance.items():
                    perf_df = pd.DataFrame(runs)
                    perf_df.to_excel(writer, sheet_name=f'Perf_{model_name}', index=False)

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

            pdf.cell(50, 10, "MAE (Validação):", 0, 0)
            pdf.cell(0, 10, f"{self.forecast_results['performance']['mae']:.2f}", 0, 1)

            pdf.cell(50, 10, "RMSE (Validação):", 0, 0)
            pdf.cell(0, 10, f"{self.forecast_results['performance']['rmse']:.2f}", 0, 1)
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
            logging.error(f"Erro ao gerar PDF: {str(e)}")
            return None


# Criação da interface
def main():
    st.title("🍞 Sistema Avançado de Previsão de Demanda - Padaria Master")

    # Inicializa o sistema
    system = AdvancedDemandForecastSystem()
    system.load_data()

    # Barra lateral
    st.sidebar.header("Menu")
    menu_options = {
        "📊 Dados": show_data_tab,
        "🔍 Análise": show_analysis_tab,
        "🔮 Previsão": show_forecast_tab,
        "📊 Visualização": show_visualization_tab,
        "📈 Desempenho": show_performance_tab,
        "📤 Exportar": show_export_tab,
        "⚙ Configurações": show_settings_tab
    }

    selected_tab = st.sidebar.radio("Navegação", list(menu_options.keys()))

    # Exibe a aba selecionada
    menu_options[selected_tab](system)


def show_data_tab(system):
    """Exibe a aba de gerenciamento de dados"""
    st.header("📊 Gerenciamento de Dados")

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
                if system.import_data(uploaded_files):
                    st.rerun()

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
        # Mostrar relatório de qualidade de dados
        if system.data_quality_report:
            st.subheader("Relatório de Qualidade dos Dados")

            col1, col2, col3 = st.columns(3)
            col1.metric("Valores Negativos", system.data_quality_report['negative_values'])
            col2.metric("Datas Faltantes", system.data_quality_report['missing_dates'])
            col3.metric("Outliers Detectados", system.data_quality_report['outliers'])

            if (system.data_quality_report['negative_values'] > 0 or
                    system.data_quality_report['missing_dates'] > 0 or
                    system.data_quality_report['outliers'] > 0):
                st.warning("Foram detectados problemas nos dados. O sistema fez ajustes automáticos.")

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


def show_analysis_tab(system):
    """Exibe a aba de análise exploratória"""
    st.header("🔍 Análise Exploratória")

    if system.data.empty:
        st.warning("Nenhum dado disponível para análise")
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
            value=min_date,
            key="analysis_start"
        )

    with col2:
        end_date = st.date_input(
            "Data Final",
            min_value=min_date,
            max_value=max_date,
            value=max_date,
            key="analysis_end"
        )

    # Filtrar dados
    filtered_data = system.data[
        (system.data['Data'] >= pd.to_datetime(start_date)) &
        (system.data['Data'] <= pd.to_datetime(end_date))
        ]

    if filtered_data.empty:
        st.warning("Nenhum dado no período selecionado")
        return

    # Tabs para diferentes análises
    tab1, tab2, tab3 = st.tabs(["Série Temporal", "Análise de Sazonalidade", "Distribuição"])

    with tab1:
        fig = px.line(
            filtered_data,
            x='Data',
            y='Unidades Vendidas',
            title='Série Temporal de Vendas',
            labels={'Unidades Vendidas': 'Unidades Vendidas', 'Data': 'Data'}
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        # Decomposição sazonal
        try:
            if len(filtered_data) >= 30:
                st.subheader("Decomposição Sazonal")
                decomposition_fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

                # Adiciona série original
                decomposition_fig.add_trace(
                    go.Scatter(x=filtered_data['Data'], y=filtered_data['Unidades Vendidas'], name='Observado'),
                    row=1, col=1
                )

                # Adiciona tendência, sazonalidade e resíduos (simplificado)
                rolling_mean = filtered_data['Unidades Vendidas'].rolling(window=7).mean()
                decomposition_fig.add_trace(
                    go.Scatter(x=filtered_data['Data'], y=rolling_mean, name='Tendência'),
                    row=2, col=1
                )

                # Sazonalidade (simplificada)
                filtered_data['DiaSemana'] = filtered_data['Data'].dt.dayofweek
                seasonal = filtered_data.groupby('DiaSemana')['Unidades Vendidas'].mean()
                seasonal_component = filtered_data['DiaSemana'].map(seasonal)
                decomposition_fig.add_trace(
                    go.Scatter(x=filtered_data['Data'], y=seasonal_component, name='Sazonalidade'),
                    row=3, col=1
                )

                # Resíduos
                residuals = filtered_data['Unidades Vendidas'] - rolling_mean - seasonal_component
                decomposition_fig.add_trace(
                    go.Scatter(x=filtered_data['Data'], y=residuals, name='Resíduo'),
                    row=4, col=1
                )

                decomposition_fig.update_layout(height=800, title_text="Decomposição Simplificada")
                st.plotly_chart(decomposition_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível realizar a decomposição: {str(e)}")

    with tab2:
        filtered_data['DiaSemana'] = filtered_data['Data'].dt.day_name()
        filtered_data['Mes'] = filtered_data['Data'].dt.month_name()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Por Dia da Semana")
            fig = px.box(
                filtered_data,
                x='DiaSemana',
                y='Unidades Vendidas',
                title='Distribuição por Dia da Semana'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Média por Dia da Semana")
            mean_by_day = filtered_data.groupby('DiaSemana')['Unidades Vendidas'].mean().reset_index()
            fig = px.bar(
                mean_by_day,
                x='DiaSemana',
                y='Unidades Vendidas',
                title='Média de Vendas por Dia da Semana'
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(filtered_data) > 60:  # Só mostra análise mensal se tiver dados suficientes
            st.subheader("Padrão Mensal")
            monthly_data = filtered_data.groupby('Mes')['Unidades Vendidas'].mean().reset_index()
            fig = px.line(
                monthly_data,
                x='Mes',
                y='Unidades Vendidas',
                title='Média de Vendas por Mês'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Histograma")
            fig = px.histogram(
                filtered_data,
                x='Unidades Vendidas',
                nbins=30,
                title='Distribuição das Vendas Diárias'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("QQ-Plot")
            qq_data = stats.probplot(filtered_data['Unidades Vendidas'], dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Dados'
            ))
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][0] * qq_data[1][1] + qq_data[1][2],
                mode='lines',
                name='Distribuição Normal'
            ))
            fig.update_layout(
                title='QQ-Plot (Comparação com Distribuição Normal)',
                xaxis_title='Quantis Teóricos',
                yaxis_title='Quantis Amostrais'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_forecast_tab(system):
    """Exibe a aba de previsão de demanda"""
    st.header("🔮 Previsão de Demanda")

    if system.data.empty:
        st.warning("Carregue dados na aba '📊 Dados' antes de executar previsões.")
        return

    if len(system.data) < 30:
        st.warning("Atenção: São recomendados pelo menos 30 dias de dados para previsões confiáveis.")

    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox(
            "Selecione o Modelo",
            ["ARIMA", "SARIMA", "Holt-Winters", "Random Forest", "XGBoost", "Prophet"],
            help="""ARIMA: Modelo estatístico para séries temporais
SARIMA: ARIMA com componente sazonal
Holt-Winters: Modelo com componentes de tendência e sazonalidade
Random Forest: Modelo de ensemble baseado em árvores
XGBoost: Modelo gradient boosting otimizado
Prophet: Modelo do Facebook para séries temporais"""
        )

    with col2:
        horizon = st.selectbox(
            "Horizonte de Previsão (dias)",
            [7, 14, 30],
            help="Número de dias no futuro para prever"
        )

    # Configurações específicas do modelo
    if model == "ARIMA":
        st.subheader("Configurações ARIMA")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write("Ordem do Modelo (p, d, q)")
            p = st.slider("p (Auto-regressivo)", 0, 5, st.session_state.model_params['ARIMA']['order'][0])
            d = st.slider("d (Diferenciação)", 0, 2, st.session_state.model_params['ARIMA']['order'][1])
            q = st.slider("q (Média móvel)", 0, 5, st.session_state.model_params['ARIMA']['order'][2])

        st.session_state.model_params['ARIMA']['order'] = (p, d, q)

        if st.button("Autoajustar ARIMA"):
            system.auto_tune_arima()

    elif model == "SARIMA":
        st.subheader("Configurações SARIMA")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Ordem do Modelo (p, d, q)")
            p = st.slider("p (Auto-regressivo)", 0, 2, st.session_state.model_params['SARIMA']['order'][0])
            d = st.slider("d (Diferenciação)", 0, 2, st.session_state.model_params['SARIMA']['order'][1])
            q = st.slider("q (Média móvel)", 0, 2, st.session_state.model_params['SARIMA']['order'][2])

        with col2:
            st.write("Ordem Sazonal (P, D, Q, m)")
            P = st.slider("P (Sazonal AR)", 0, 2, st.session_state.model_params['SARIMA']['seasonal_order'][0])
            D = st.slider("D (Sazonal Diff)", 0, 2, st.session_state.model_params['SARIMA']['seasonal_order'][1])
            Q = st.slider("Q (Sazonal MA)", 0, 2, st.session_state.model_params['SARIMA']['seasonal_order'][2])
            m = st.slider("m (Período)", 7, 30, st.session_state.model_params['SARIMA']['seasonal_order'][3], step=7)

        st.session_state.model_params['SARIMA']['order'] = (p, d, q)
        st.session_state.model_params['SARIMA']['seasonal_order'] = (P, D, Q, m)

        if st.button("Autoajustar SARIMA"):
            system.auto_tune_sarima()

    elif model == "Holt-Winters":
        st.subheader("Configurações Holt-Winters")
        col1, col2 = st.columns(2)

        with col1:
            trend = st.selectbox("Tendência", ['add', 'mul'],
                                 index=0 if st.session_state.model_params['Holt-Winters']['trend'] == 'add' else 1,
                                 help="Tipo de componente de tendência")
            seasonal = st.selectbox("Sazonalidade", ['add', 'mul'],
                                    index=0 if st.session_state.model_params['Holt-Winters'][
                                                   'seasonal'] == 'add' else 1,
                                    help="Tipo de componente sazonal")
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

    elif model == "Random Forest":
        st.subheader("Configurações Random Forest")
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
                value=5 if st.session_state.model_params['Random Forest']['max_depth'] is None else
                st.session_state.model_params['Random Forest']['max_depth'],
                help="None para profundidade ilimitada"
            )

        st.session_state.model_params['Random Forest'] = {
            'n_estimators': n_estimators,
            'max_depth': None if max_depth == 0 else max_depth
        }

    elif model == "XGBoost":
        st.subheader("Configurações XGBoost")
        col1, col2 = st.columns(2)

        with col1:
            n_estimators = st.number_input(
                "Número de Árvores",
                min_value=10,
                max_value=500,
                value=st.session_state.model_params['XGBoost']['n_estimators']
            )
            learning_rate = st.number_input(
                "Taxa de Aprendizado",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                value=st.session_state.model_params['XGBoost']['learning_rate']
            )
        with col2:
            max_depth = st.number_input(
                "Profundidade Máxima",
                min_value=1,
                max_value=20,
                value=st.session_state.model_params['XGBoost']['max_depth']
            )

        st.session_state.model_params['XGBoost'] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }

    else:  # Prophet
        st.subheader("Configurações Prophet")
        col1, col2 = st.columns(2)

        with col1:
            changepoint_prior_scale = st.slider(
                "Sensibilidade a Mudanças",
                min_value=0.001,
                max_value=0.5,
                value=st.session_state.model_params['Prophet']['changepoint_prior_scale'],
                step=0.001,
                help="Controla a flexibilidade da tendência"
            )
        with col2:
            seasonality_prior_scale = st.slider(
                "Força da Sazonalidade",
                min_value=0.01,
                max_value=50.0,
                value=st.session_state.model_params['Prophet']['seasonality_prior_scale'],
                step=0.1,
                help="Controla a força dos componentes sazonais"
            )

        st.session_state.model_params['Prophet'] = {
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale
        }

        if st.button("Autoajustar Prophet"):
            system.auto_tune_prophet()

    if st.button("Executar Previsão", type="primary"):
        if system.run_forecast(model, horizon):
            st.rerun()

    if system.forecast_results:
        st.divider()
        st.subheader("Resultados da Previsão")

        # Gráfico de previsão interativo
        fig = go.Figure()

        # Adiciona dados históricos
        fig.add_trace(go.Scatter(
            x=system.data['Data'],
            y=system.data['Unidades Vendidas'],
            mode='lines',
            name='Histórico',
            line=dict(color='blue')
        ))

        # Adiciona previsão
        forecast_dates = pd.date_range(
            start=system.forecast_results['last_date'] + timedelta(days=1),
            periods=system.forecast_results['horizon']
        )
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=system.forecast_results['forecast'],
            mode='lines+markers',
            name=f'Previsão ({system.forecast_results["model"]})',
            line=dict(color='red', dash='dash')
        ))

        # Adiciona intervalo de confiança (simulado)
        if system.forecast_results['model'] in ['ARIMA', 'SARIMA', 'Holt-Winters', 'Prophet']:
            std_dev = np.std(system.data['Unidades Vendidas'])
            upper_bound = system.forecast_results['forecast'] + 1.96 * std_dev
            lower_bound = system.forecast_results['forecast'] - 1.96 * std_dev

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.1)',
                name='Intervalo de Confiança (95%)'
            ))

        fig.update_layout(
            title=f"Previsão de Demanda - {system.forecast_results['model']}",
            xaxis_title="Data",
            yaxis_title="Unidades Vendidas",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabela de previsões com detalhes
        forecast_df = pd.DataFrame({
            'Data': forecast_dates,
            'Unidades Previstas': system.forecast_results['forecast'],
            'Dia da Semana': forecast_dates.day_name(),
            'MAE (Validação)': system.forecast_results['performance']['mae'],
            'RMSE (Validação)': system.forecast_results['performance']['rmse']
        })

        st.dataframe(
            forecast_df.style.format({
                'Unidades Previstas': '{:.1f}',
                'MAE (Validação)': '{:.1f}',
                'RMSE (Validação)': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )


def show_visualization_tab(system):
    """Exibe a aba de visualização de dados"""
    st.header("📊 Visualização de Dados")

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
            value=min_date,
            key="viz_start"
        )

    with col2:
        end_date = st.date_input(
            "Data Final",
            min_value=min_date,
            max_value=max_date,
            value=max_date,
            key="viz_end"
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
        fig = px.line(
            filtered_data,
            x='Data',
            y='Unidades Vendidas',
            title='Vendas por Data',
            labels={'Unidades Vendidas': 'Unidades Vendidas', 'Data': 'Data'}
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        filtered_data['DiaSemana'] = filtered_data['Data'].dt.day_name()

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Distribuição por Dia da Semana", "Média de Vendas por Dia"))

        # Boxplot
        fig.add_trace(
            go.Box(
                x=filtered_data['DiaSemana'],
                y=filtered_data['Unidades Vendidas'],
                name='Distribuição'
            ),
            row=1, col=1
        )

        # Média por dia
        mean_by_day = filtered_data.groupby('DiaSemana')['Unidades Vendidas'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=mean_by_day['DiaSemana'],
                y=mean_by_day['Unidades Vendidas'],
                name='Média'
            ),
            row=1, col=2
        )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_performance_tab(system):
    """Exibe a aba de desempenho dos modelos"""
    st.header("📈 Desempenho dos Modelos")

    if not system.model_performance:
        st.warning(
            "Nenhum modelo foi executado ainda. Execute previsões na aba '🔮 Previsão' para ver métricas de desempenho.")
        return

    # Mostrar métricas de todos os modelos
    st.subheader("Comparação de Modelos")

    # Cria dataframe com todas as execuções
    performance_data = []
    for model_name, runs in system.model_performance.items():
        for run in runs:
            performance_data.append({
                'Modelo': model_name,
                'Data Execução': run['date'].strftime('%Y-%m-%d %H:%M'),
                'Horizonte': run['horizon'],
                'MAE': run['mae'],
                'RMSE': run['rmse']
            })

    if not performance_data:
        st.warning("Nenhuma métrica de desempenho disponível")
        return

    perf_df = pd.DataFrame(performance_data)

    # Melhor modelo por horizonte
    st.subheader("Melhor Modelo por Horizonte")
    for horizon in [7, 14, 30]:
        horizon_df = perf_df[perf_df['Horizonte'] == horizon]
        if not horizon_df.empty:
            best_model = horizon_df.loc[horizon_df['MAE'].idxmin()]
            st.write(f"**Horizonte {horizon} dias:** {best_model['Modelo']} (MAE: {best_model['MAE']:.1f})")

    # Gráfico de evolução do MAE
    st.subheader("Evolução do Erro (MAE) por Modelo")
    fig = px.line(
        perf_df,
        x='Data Execução',
        y='MAE',
        color='Modelo',
        facet_col='Horizonte',
        title='Evolução do MAE por Horizonte de Previsão',
        labels={'MAE': 'Erro Absoluto Médio', 'Data Execução': 'Data da Execução'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela detalhada
    st.subheader("Histórico de Execuções")
    st.dataframe(
        perf_df.sort_values('Data Execução', ascending=False),
        use_container_width=True,
        hide_index=True
    )


def show_export_tab(system):
    """Exibe a aba de exportação de dados"""
    st.header("📤 Exportar Dados")

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

    # Exportar modelo treinado
    if system.forecast_results:
        st.subheader("Exportar Modelo Treinado")
        model_name = system.forecast_results['model']

        if st.button(f"Salvar Modelo {model_name}"):
            try:
                # Simulação - em um sistema real, salvaríamos o modelo real
                model_data = {
                    'model_name': model_name,
                    'parameters': st.session_state.model_params[model_name],
                    'last_training_date': datetime.now(),
                    'performance': system.forecast_results['performance']
                }

                output = BytesIO()
                joblib.dump(model_data, output)

                st.download_button(
                    label=f"Baixar Modelo {model_name}",
                    data=output.getvalue(),
                    file_name=f"modelo_{model_name.lower()}.joblib",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Falha ao exportar modelo: {str(e)}")


def show_settings_tab(system):
    """Exibe a aba de configurações"""
    st.header("⚙ Configurações do Sistema")

    st.subheader("Configurações de Modelos")

    with st.expander("Configurações ARIMA"):
        st.write("Parâmetros atuais:", st.session_state.model_params['ARIMA'])
        if st.button("Redefinir ARIMA"):
            st.session_state.model_params['ARIMA'] = {'order': (5, 1, 0)}
            st.rerun()

    with st.expander("Configurações SARIMA"):
        st.write("Parâmetros atuais:", st.session_state.model_params['SARIMA'])
        if st.button("Redefinir SARIMA"):
            st.session_state.model_params['SARIMA'] = {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)}
            st.rerun()

    with st.expander("Configurações Holt-Winters"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Holt-Winters'])
        if st.button("Redefinir Holt-Winters"):
            st.session_state.model_params['Holt-Winters'] = {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7}
            st.rerun()

    with st.expander("Configurações Random Forest"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Random Forest'])
        if st.button("Redefinir Random Forest"):
            st.session_state.model_params['Random Forest'] = {'n_estimators': 100, 'max_depth': None}
            st.rerun()

    with st.expander("Configurações XGBoost"):
        st.write("Parâmetros atuais:", st.session_state.model_params['XGBoost'])
        if st.button("Redefinir XGBoost"):
            st.session_state.model_params['XGBoost'] = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
            st.rerun()

    with st.expander("Configurações Prophet"):
        st.write("Parâmetros atuais:", st.session_state.model_params['Prophet'])
        if st.button("Redefinir Prophet"):
            st.session_state.model_params['Prophet'] = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
            st.rerun()

    st.subheader("Configurações do Sistema")
    cache_settings = st.checkbox("Usar cache para melhor performance", value=True)
    auto_backup = st.checkbox("Ativar backup automático", value=True)

    st.subheader("Sobre o Sistema")
    st.write("""
        **Sistema Avançado de Previsão de Demanda**  
        Versão 3.0  
        Desenvolvido com Streamlit, Plotly e Scikit-learn  
        © 2023 - Todos os direitos reservados
    """)


if __name__ == "__main__":
    main()