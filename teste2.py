import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
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
from typing import Dict, List, Optional, Tuple, Union

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
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas', 'Produto', 'Categoria'])
        self.forecast_results = None
        self.forecast_history = []
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
        if 'forecast_history' not in st.session_state:
            st.session_state.forecast_history = []
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
        self.forecast_history = st.session_state.forecast_history

    def save_data(self):
        """Salva dados na sessão e cria backup"""
        st.session_state.data = self.data
        st.session_state.forecast_results = self.forecast_results
        st.session_state.forecast_history = self.forecast_history

        # Criar backup (mantém últimos 5)
        if len(st.session_state.data_backups) >= 5:
            st.session_state.data_backups.pop(0)
        st.session_state.data_backups.append({
            'data': self.data.copy(),
            'forecast_results': self.forecast_results.copy() if self.forecast_results else None,
            'forecast_history': self.forecast_history.copy()
        })

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processamento de dados com cache"""
        processed = data.copy()
        processed['Data'] = pd.to_datetime(processed['Data'])
        processed = processed.sort_values('Data')

        # Garantir que todas as colunas necessárias existam
        if 'Categoria' not in processed.columns:
            processed['Categoria'] = 'Geral'

        return processed

    def import_data(self, uploaded_files):
        """Importa dados de múltiplos arquivos com tratamento robusto de formatos"""
        try:
            new_dfs = []

            for uploaded_file in uploaded_files:
                # Leitura do arquivo
                if uploaded_file.name.endswith('.xlsx'):
                    new_data = pd.read_excel(uploaded_file)
                else:
                    # Para CSV, detecta automaticamente separador e encoding
                    new_data = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')

                # Verificação das colunas obrigatórias
                required_cols = ['Data', 'Unidades Vendidas']
                if not all(col in new_data.columns for col in required_cols):
                    raise ValueError(f"Arquivo {uploaded_file.name} deve conter colunas 'Data' e 'Unidades Vendidas'")

                # Tratamento de datas - múltiplos formatos suportados
                new_data['Data'] = pd.to_datetime(
                    new_data['Data'],
                    dayfirst=True,  # Prioriza formato DD/MM/YYYY
                    yearfirst=False,  # Só considera YYYY primeiro se dayfirst falhar
                    format='mixed',  # Aceita múltiplos formatos
                    errors='coerce'  # Converte falhas para NaT
                )

                # Remove linhas com datas inválidas
                if new_data['Data'].isna().any():
                    invalid_dates = new_data[new_data['Data'].isna()]
                    st.warning(
                        f"Removidas {len(invalid_dates)} linhas com datas inválidas no arquivo {uploaded_file.name}")
                    new_data = new_data.dropna(subset=['Data'])

                # Tratamento de valores numéricos (suporte a vírgula decimal)
                if new_data['Unidades Vendidas'].dtype == object:
                    new_data['Unidades Vendidas'] = (
                        new_data['Unidades Vendidas']
                        .astype(str)
                        .str.replace(',', '.')
                        .astype(float)
                    )

                # Preenchimento de colunas opcionais
                if 'Produto' not in new_data.columns:
                    new_data['Produto'] = 'Padaria Geral'
                if 'Categoria' not in new_data.columns:
                    new_data['Categoria'] = 'Geral'

                new_dfs.append(new_data)

            if not new_dfs:
                return False

            # Consolida todos os dataframes
            new_data = pd.concat(new_dfs, ignore_index=True)

            # Processamento adicional
            new_data = self._process_data(new_data)
            new_data = self.handle_missing_dates(new_data)
            new_data = self.detect_outliers(new_data)

            # Verificação de duplicatas
            dup_cols = ['Data', 'Produto'] if 'Produto' in new_data.columns else ['Data']
            if new_data.duplicated(subset=dup_cols).any():
                st.warning(f"Removidas {new_data.duplicated(subset=dup_cols).sum()} duplicatas")
                new_data = new_data.drop_duplicates(subset=dup_cols, keep='last')

            # Atualiza os dados do sistema
            self.data = new_data
            self.save_data()
            st.success(f"Dados importados com sucesso! {len(self.data)} registros carregados.")
            return True

        except Exception as e:
            st.error(f"Falha na importação: {str(e)}")
            logging.error(f"Import error: {str(e)}")
            return False

    def handle_missing_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preenche datas faltantes com interpolação"""
        if data.empty:
            return data

        # Criar range completo de datas
        date_range = pd.date_range(
            start=data['Data'].min(),
            end=data['Data'].max(),
            freq='D'
        )

        # Para cada combinação de produto e categoria, preencher datas faltantes
        group_cols = []
        if 'Produto' in data.columns:
            group_cols.append('Produto')
        if 'Categoria' in data.columns:
            group_cols.append('Categoria')

        if not group_cols:
            # Sem agrupamento por produto/categoria
            data = data.set_index('Data').reindex(date_range)
            data['Unidades Vendidas'] = data['Unidades Vendidas'].interpolate(
                method='linear',
                limit_direction='both'
            ).fillna(0)
            return data.reset_index().rename(columns={'index': 'Data'})
        else:
            # Com agrupamento
            filled_dfs = []
            groups = data.groupby(group_cols)

            for (group_key), group_data in groups:
                if isinstance(group_key, str):
                    group_key = [group_key]

                # Reindexar para todas as datas
                group_data = group_data.set_index('Data').reindex(date_range)

                # Manter os valores de grupo
                for col, val in zip(group_cols, group_key):
                    group_data[col] = val

                # Interpolar valores faltantes
                group_data['Unidades Vendidas'] = group_data['Unidades Vendidas'].interpolate(
                    method='linear',
                    limit_direction='both'
                ).fillna(0)  # Preencher com 0 se não puder interpolar

                filled_dfs.append(group_data.reset_index().rename(columns={'index': 'Data'}))

            return pd.concat(filled_dfs, ignore_index=True)

    def detect_outliers(self, data: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """Detecta e trata outliers usando Z-Score"""
        if data.empty:
            return data

        data = data.copy()

        # Tratar outliers por grupo (produto/categoria) se existirem
        group_cols = []
        if 'Produto' in data.columns and len(data['Produto'].unique()) > 1:
            group_cols.append('Produto')
        if 'Categoria' in data.columns and len(data['Categoria'].unique()) > 1:
            group_cols.append('Categoria')

        if not group_cols:
            # Sem agrupamento
            z_scores = np.abs(stats.zscore(data['Unidades Vendidas']))
            outliers = z_scores > threshold

            if outliers.any():
                st.warning(f"Detectados {outliers.sum()} outliers. Eles serão substituídos pela mediana.")
                median_val = data['Unidades Vendidas'].median()
                data.loc[outliers, 'Unidades Vendidas'] = median_val
        else:
            # Com agrupamento
            groups = data.groupby(group_cols)
            processed_groups = []

            for _, group_data in groups:
                z_scores = np.abs(stats.zscore(group_data['Unidades Vendidas']))
                outliers = z_scores > threshold

                if outliers.any():
                    median_val = group_data['Unidades Vendidas'].median()
                    group_data.loc[outliers, 'Unidades Vendidas'] = median_val

                processed_groups.append(group_data)

            data = pd.concat(processed_groups, ignore_index=True)

        return data

    def add_manual_data(self, date: str, units: float, product: str, category: str) -> bool:
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
                new_data = pd.DataFrame([[date, units, product, category]],
                                        columns=['Data', 'Unidades Vendidas', 'Produto', 'Categoria'])
                self.data = pd.concat([self.data, new_data], ignore_index=True)

            self.data = self._process_data(self.data)
            self.save_data()
            st.success("Dados adicionados com sucesso!")
            return True

        except ValueError as e:
            st.error(f"Formato inválido: {str(e)}")
            logging.error(f"Manual data error: {str(e)}")
            return False

    def clear_data(self) -> None:
        """Limpa todos os dados"""
        self.data = pd.DataFrame(columns=['Data', 'Unidades Vendidas', 'Produto', 'Categoria'])
        self.forecast_results = None
        self.forecast_history = []
        self.save_data()
        st.success("Dados limpos com sucesso!")

    def run_forecast(self, model_name: str, horizon: int, product: Optional[str] = None,
                     category: Optional[str] = None, **kwargs) -> bool:
        """Executa a previsão de demanda com validação cruzada"""
        try:
            # Filtrar por produto e categoria se especificado
            filter_condition = True
            if product and 'Produto' in self.data.columns:
                filter_condition &= (self.data['Produto'] == product)
            if category and 'Categoria' in self.data.columns:
                filter_condition &= (self.data['Categoria'] == category)

            filtered_data = self.data[filter_condition]

            if filtered_data.empty:
                st.error("Nenhum dado encontrado para os filtros aplicados")
                return False

            ts_data = filtered_data.set_index('Data')['Unidades Vendidas']

            if len(ts_data) < 30:
                st.warning("Dados insuficientes para previsão confiável. Recomendado pelo menos 30 pontos.")

            # Configurar progresso
            progress_bar = st.progress(0)
            status_text = st.empty()

            if model_name == "ARIMA":
                status_text.text("Ajustando modelo ARIMA...")
                order = st.session_state.model_params['ARIMA']['order']
                forecast, forecast_min, forecast_max = self.arima_forecast(ts_data, horizon, order)
                progress_bar.progress(100)

            elif model_name == "SARIMA":
                status_text.text("Ajustando modelo SARIMA...")
                order = st.session_state.model_params['SARIMA']['order']
                seasonal_order = st.session_state.model_params['SARIMA']['seasonal_order']
                forecast, forecast_min, forecast_max = self.sarima_forecast(ts_data, horizon, order, seasonal_order)
                progress_bar.progress(100)

            elif model_name == "Holt-Winters":
                status_text.text("Ajustando modelo Holt-Winters...")
                params = st.session_state.model_params['Holt-Winters']
                forecast, forecast_min, forecast_max = self.holt_winters_forecast(ts_data, horizon, **params)
                progress_bar.progress(100)

            elif model_name == "Prophet":
                status_text.text("Ajustando modelo Prophet...")
                params = st.session_state.model_params['Prophet']
                forecast, forecast_min, forecast_max = self.prophet_forecast(ts_data, horizon, **params)
                progress_bar.progress(100)

            elif model_name == "Random Forest":
                status_text.text("Treinando Random Forest com validação cruzada...")
                params = st.session_state.model_params['Random Forest']
                forecast, forecast_min, forecast_max = self.ml_forecast(ts_data, horizon, model='rf', **params)
                progress_bar.progress(100)

            elif model_name == "XGBoost":
                status_text.text("Treinando XGBoost com validação cruzada...")
                params = st.session_state.model_params['XGBoost']
                forecast, forecast_min, forecast_max = self.ml_forecast(ts_data, horizon, model='xgb', **params)
                progress_bar.progress(100)

            else:
                raise ValueError(f"Modelo desconhecido: {model_name}")

            # Calcular métricas de erro se houver dados suficientes
            error_metrics = {}
            if len(ts_data) > horizon:
                train_data = ts_data.iloc[:-horizon]
                test_data = ts_data.iloc[-horizon:]

                # Re-treinar o modelo com dados de treino
                if model_name == "ARIMA":
                    model = ARIMA(train_data, order=order)
                    model_fit = model.fit()
                    test_forecast = model_fit.forecast(steps=horizon)
                elif model_name == "SARIMA":
                    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
                    model_fit = model.fit(disp=False)
                    test_forecast = model_fit.forecast(steps=horizon)
                elif model_name == "Holt-Winters":
                    model = ExponentialSmoothing(
                        train_data,
                        trend=params.get('trend', 'add'),
                        seasonal=params.get('seasonal', 'add'),
                        seasonal_periods=params.get('seasonal_periods', 7)
                    )
                    model_fit = model.fit()
                    test_forecast = model_fit.forecast(horizon)
                elif model_name == "Prophet":
                    df = pd.DataFrame({
                        'ds': train_data.index,
                        'y': train_data.values
                    })
                    model = Prophet(**params)
                    model.fit(df)
                    future = model.make_future_dataframe(periods=horizon)
                    test_forecast = model.predict(future).tail(horizon)['yhat'].values
                elif model_name in ["Random Forest", "XGBoost"]:
                    test_forecast, _, _ = self.ml_forecast(train_data, horizon,
                                                           model='rf' if model_name == "Random Forest" else 'xgb',
                                                           **params)

                # Calcular métricas
                error_metrics = {
                    'MAPE': mean_absolute_percentage_error(test_data, test_forecast) * 100,
                    'RMSE': np.sqrt(mean_squared_error(test_data, test_forecast)),
                    'Accuracy': max(0, 100 - mean_absolute_percentage_error(test_data, test_forecast) * 100)
                }

            # Salvar resultados
            forecast_dates = pd.date_range(
                start=ts_data.index[-1] + timedelta(days=1),
                periods=horizon
            )

            forecast_df = pd.DataFrame({
                'Data': forecast_dates,
                'Unidades Previstas': forecast,
                'Previsão Mínima': forecast_min if forecast_min is not None else forecast,
                'Previsão Máxima': forecast_max if forecast_max is not None else forecast,
                'Modelo': model_name,
                'Produto': product if product else 'Geral',
                'Categoria': category if category else 'Geral'
            })

            self.forecast_results = {
                'model': model_name,
                'horizon': horizon,
                'forecast': forecast_df,
                'last_date': ts_data.index[-1],
                'execution_date': datetime.now(),
                'product': product,
                'category': category,
                'error_metrics': error_metrics if error_metrics else None
            }

            # Adicionar ao histórico (mantém últimos 10)
            if len(self.forecast_history) >= 10:
                self.forecast_history.pop(0)
            self.forecast_history.append(self.forecast_results.copy())

            self.save_data()
            status_text.text("Previsão concluída com sucesso!")
            st.success("Previsão realizada com sucesso!")
            return True

        except Exception as e:
            st.error(f"Falha na previsão: {str(e)}")
            logging.error(f"Forecast error: {str(e)}")
            return False

    def arima_forecast(self, ts_data: pd.Series, horizon: int, order: Tuple[int, int, int]) -> Tuple[
        np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Previsão com modelo ARIMA"""
        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)

        # Obter intervalos de confiança
        conf_int = model_fit.get_forecast(steps=horizon).conf_int()
        forecast_min = conf_int.iloc[:, 0].values
        forecast_max = conf_int.iloc[:, 1].values

        return forecast.values, forecast_min, forecast_max

    def sarima_forecast(self, ts_data: pd.Series, horizon: int, order: Tuple[int, int, int],
                        seasonal_order: Tuple[int, int, int, int]) -> Tuple[
        np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Previsão com modelo SARIMA"""
        model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=horizon)

        # Obter intervalos de confiança
        conf_int = model_fit.get_forecast(steps=horizon).conf_int()
        forecast_min = conf_int.iloc[:, 0].values
        forecast_max = conf_int.iloc[:, 1].values

        return forecast.values, forecast_min, forecast_max

    def holt_winters_forecast(self, ts_data: pd.Series, horizon: int, trend: str, seasonal: str,
                              seasonal_periods: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Previsão com modelo Holt-Winters"""
        model = ExponentialSmoothing(
            ts_data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(horizon)

        # Holt-Winters não fornece intervalos de confiança diretamente
        return forecast.values, None, None

    def prophet_forecast(self, ts_data: pd.Series, horizon: int, **params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Previsão com Facebook Prophet"""
        df = pd.DataFrame({
            'ds': ts_data.index,
            'y': ts_data.values
        })

        model = Prophet(**params, interval_width=0.95)  # 95% de intervalo de confiança
        model.fit(df)

        future = model.make_future_dataframe(periods=horizon)
        forecast_df = model.predict(future).tail(horizon)

        return (
            forecast_df['yhat'].values,
            forecast_df['yhat_lower'].values,
            forecast_df['yhat_upper'].values
        )

    def ml_forecast(self, ts_data: pd.Series, horizon: int, model: str = 'rf',
                    **params) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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

        # ML models não fornecem intervalos de confiança diretamente
        return (
            np.array(forecasts),
            None,
            None
        )

    def auto_tune_model(self, model_name: str) -> None:
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

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calcula estatísticas com cache"""
        if data.empty:
            return {
                'total': 0,
                'start_date': None,
                'end_date': None,
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0
            }

        # Garantir que as colunas necessárias existam
        if 'Unidades Vendidas' not in data.columns:
            data['Unidades Vendidas'] = 0

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

        if 'Produto' in data.columns:
            stats['products'] = data['Produto'].nunique()
        if 'Categoria' in data.columns:
            stats['categories'] = data['Categoria'].nunique()

        return stats
    def export_to_excel(self) -> Optional[BytesIO]:
        """Exporta dados para Excel com todas as informações"""
        if self.data.empty:
            st.warning("Nenhum dado para exportar")
            return None

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Dados históricos
            self.data.to_excel(writer, sheet_name='Dados Históricos', index=False)

            # Previsões atuais
            if self.forecast_results:
                forecast_df = self.forecast_results['forecast']
                forecast_df.to_excel(writer, sheet_name='Previsões Atuais', index=False)

                # Adicionar métricas de erro se existirem
                if self.forecast_results.get('error_metrics'):
                    metrics_df = pd.DataFrame.from_dict(
                        self.forecast_results['error_metrics'],
                        orient='index',
                        columns=['Valor']
                    )
                    metrics_df.to_excel(writer, sheet_name='Métricas de Erro')

            # Histórico de previsões
            if self.forecast_history:
                history_data = []
                for i, forecast in enumerate(self.forecast_history, 1):
                    forecast_df = forecast['forecast'].copy()
                    forecast_df['Execução'] = forecast['execution_date'].strftime('%Y-%m-%d %H:%M')
                    forecast_df['Modelo'] = forecast['model']
                    history_data.append(forecast_df)

                pd.concat(history_data).to_excel(writer, sheet_name='Histórico Previsões', index=False)

        return output

    def export_to_pdf(self) -> Optional[BytesIO]:
        """Exporta relatório para PDF com gráficos e todas as informações"""
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

            if self.forecast_results.get('product'):
                pdf.cell(50, 10, "Produto:", 0, 0)
                pdf.cell(0, 10, self.forecast_results['product'], 0, 1)

            if self.forecast_results.get('category'):
                pdf.cell(50, 10, "Categoria:", 0, 0)
                pdf.cell(0, 10, self.forecast_results['category'], 0, 1)

            pdf.ln(10)

            # Estatísticas
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Estatísticas da Previsão:", 0, 1)
            pdf.set_font("Arial", size=10)

            forecast_data = self.forecast_results['forecast']['Unidades Previstas']
            stats = [
                ("Média diária:", f"{forecast_data.mean():.2f} unidades"),
                ("Mínimo diário:", f"{forecast_data.min():.2f} unidades"),
                ("Máximo diário:", f"{forecast_data.max():.2f} unidades"),
                ("Total previsto:", f"{forecast_data.sum():.2f} unidades")
            ]

            # Adicionar métricas de erro se existirem
            if self.forecast_results.get('error_metrics'):
                stats.extend([
                    ("MAPE (Erro %):", f"{self.forecast_results['error_metrics']['MAPE']:.2f}%"),
                    ("RMSE:", f"{self.forecast_results['error_metrics']['RMSE']:.2f}"),
                    ("Taxa de Acerto:", f"{self.forecast_results['error_metrics']['Accuracy']:.2f}%")
                ])

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
            pdf.cell(30, 10, "Dia Semana", 1, 0, 'C')
            pdf.cell(40, 10, "Unidades Previstas", 1, 0, 'C')
            pdf.cell(40, 10, "Previsão Mínima", 1, 0, 'C')
            pdf.cell(40, 10, "Previsão Máxima", 1, 1, 'C')

            # Dados da tabela
            for _, row in self.forecast_results['forecast'].iterrows():
                pdf.cell(40, 10, row['Data'].strftime('%d/%m/%Y'), 1, 0, 'C')
                pdf.cell(30, 10, row['Data'].strftime('%A'), 1, 0, 'C')
                pdf.cell(40, 10, f"{row['Unidades Previstas']:.2f}", 1, 0, 'C')
                pdf.cell(40, 10, f"{row['Previsão Mínima']:.2f}", 1, 0, 'C')
                pdf.cell(40, 10, f"{row['Previsão Máxima']:.2f}", 1, 1, 'C')

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

    def get_product_list(self) -> List[str]:
        """Retorna lista de produtos únicos"""
        if 'Produto' not in self.data.columns:
            return []
        return sorted(self.data['Produto'].unique().tolist())

    def get_category_list(self) -> List[str]:
        """Retorna lista de categorias únicas"""
        if 'Categoria' not in self.data.columns:
            return []
        return sorted(self.data['Categoria'].unique().tolist())


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


def show_data_tab(system: AdvancedDemandForecastSystem) -> None:
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

                categories = system.get_category_list()
                new_category = st.text_input("Nova Categoria (ou selecione abaixo)")

                if categories:
                    selected_category = st.selectbox(
                        "Categoria Existente",
                        options=categories,
                        index=0
                    )
                    category = new_category if new_category else selected_category
                else:
                    category = new_category if new_category else "Geral"

            if st.form_submit_button("Adicionar Dados"):
                if system.add_manual_data(date, units, product, category):
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

            categories = system.get_category_list()
            if categories:
                selected_categories = st.multiselect(
                    "Categorias",
                    options=categories,
                    default=categories
                )
            else:
                selected_categories = []

        # Aplicar filtros
        filtered_data = system.data[
            (system.data['Data'] >= pd.to_datetime(start_date)) &
            (system.data['Data'] <= pd.to_datetime(end_date))
            ]

        if selected_products:
            filtered_data = filtered_data[filtered_data['Produto'].isin(selected_products)]
        if selected_categories:
            filtered_data = filtered_data[filtered_data['Categoria'].isin(selected_categories)]

        # Mostrar dados
        st.dataframe(filtered_data, use_container_width=True, hide_index=True)

        # Mostrar estatísticas básicas
        stats = system.calculate_statistics(filtered_data)

        st.write(f"**Total de registros:** {stats['total']}")
        st.write(f"**Período:** {stats['start_date'].date()} a {stats['end_date'].date()}")
        st.write(f"**Média diária:** {stats['mean']:.2f} unidades")
        st.write(f"**Mediana diária:** {stats['median']:.2f} unidades")

        if 'products' in stats:
            st.write(f"**Número de produtos:** {stats['products']}")
        if 'categories' in stats:
            st.write(f"**Número de categorias:** {stats['categories']}")

        if len(filtered_data) < 30:
            st.warning("São necessários pelo menos 30 dias de dados para previsões confiáveis.")

        if st.button("Limpar Todos os Dados", type="primary"):
            system.clear_data()
            st.rerun()
    else:
        st.info("Nenhum dado carregado. Importe arquivos ou insira dados manualmente.")


def show_forecast_tab(system: AdvancedDemandForecastSystem) -> None:
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

    # Seleção de produto e categoria se houver múltiplos
    products = system.get_product_list()
    categories = system.get_category_list()

    if products or categories:
        cols = st.columns(2)

        with cols[0]:
            if products:
                product = st.selectbox(
                    "Produto para Previsão",
                    options=["Todos"] + products,
                    index=0
                )
                product = None if product == "Todos" else product
            else:
                product = None

        with cols[1]:
            if categories:
                category = st.selectbox(
                    "Categoria para Previsão",
                    options=["Todos"] + categories,
                    index=0
                )
                category = None if category == "Todos" else category
            else:
                category = None
    else:
        product = None
        category = None

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
            if system.run_forecast(model, horizon, product, category):
                st.rerun()

    if system.forecast_results:
        st.divider()
        st.subheader("Resultados da Previsão")

        # Mostrar métricas de erro se existirem
        if system.forecast_results.get('error_metrics'):
            st.write("**Métricas de Avaliação da Previsão**")

            cols = st.columns(3)
            cols[0].metric("MAPE (Erro %)", f"{system.forecast_results['error_metrics']['MAPE']:.2f}%")
            cols[1].metric("RMSE", f"{system.forecast_results['error_metrics']['RMSE']:.2f}")
            cols[2].metric("Taxa de Acerto", f"{system.forecast_results['error_metrics']['Accuracy']:.2f}%")

        # Dados para o gráfico
        filter_condition = True
        if product and 'Produto' in system.data.columns:
            filter_condition &= (system.data['Produto'] == product)
        if category and 'Categoria' in system.data.columns:
            filter_condition &= (system.data['Categoria'] == category)

        historical_data = system.data[filter_condition]
        forecast_df = system.forecast_results['forecast']

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
            x=forecast_df['Data'],
            y=forecast_df['Unidades Previstas'],
            mode='lines',
            name=f'Previsão ({system.forecast_results["model"]})',
            line=dict(color='red', dash='dash')
        ))

        # Adicionar intervalo de confiança se disponível
        if 'Previsão Mínima' in forecast_df.columns and 'Previsão Máxima' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df['Data'], forecast_df['Data'][::-1]]),
                y=pd.concat([forecast_df['Previsão Máxima'], forecast_df['Previsão Mínima'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line_color='rgba(255,255,255,0)',
                name='Intervalo de Confiança',
                showlegend=True
            ))

        # Configurar layout
        title = f"Previsão de Demanda - {system.forecast_results['model']}"
        if product:
            title += f" - {product}"
        if category:
            title += f" ({category})"

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
        st.subheader("Detalhes da Previsão")

        display_df = forecast_df.copy()
        display_df['Dia da Semana'] = display_df['Data'].dt.day_name()

        st.dataframe(
            display_df.style.format({
                'Unidades Previstas': '{:.2f}',
                'Previsão Mínima': '{:.2f}',
                'Previsão Máxima': '{:.2f}'
            }),
            column_order=['Data', 'Dia da Semana', 'Unidades Previstas', 'Previsão Mínima', 'Previsão Máxima'],
            use_container_width=True,
            hide_index=True
        )


def show_visualization_tab(system: AdvancedDemandForecastSystem) -> None:
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

        categories = system.get_category_list()
        if categories:
            selected_categories = st.multiselect(
                "Categorias",
                options=categories,
                default=categories[:min(3, len(categories))] if categories else []
            )
        else:
            selected_categories = []

    # Filtrar dados
    filtered_data = system.data[
        (system.data['Data'] >= pd.to_datetime(start_date)) &
        (system.data['Data'] <= pd.to_datetime(end_date))
        ]

    if selected_products:
        filtered_data = filtered_data[filtered_data['Produto'].isin(selected_products)]
    if selected_categories:
        filtered_data = filtered_data[filtered_data['Categoria'].isin(selected_categories)]

    if filtered_data.empty:
        st.warning("Nenhum dado no período selecionado")
        return

    # Gráficos interativos
    tab1, tab2, tab3, tab4 = st.tabs(["Série Temporal", "Análise Sazonal", "Comparação de Produtos", "Distribuição"])

    with tab1:
        group_col = None
        if selected_products and len(selected_products) > 1:
            group_col = 'Produto'
        elif selected_categories and len(selected_categories) > 1:
            group_col = 'Categoria'

        if group_col:
            fig = px.line(
                filtered_data,
                x='Data',
                y='Unidades Vendidas',
                color=group_col,
                title=f"Vendas por Data e {group_col}",
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
            showlegend=True if group_col else False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        filtered_data['DiaSemana'] = filtered_data['Data'].dt.day_name()
        filtered_data['Mes'] = filtered_data['Data'].dt.month_name()

        group_col = None
        if selected_products and len(selected_products) > 1:
            group_col = 'Produto'
        elif selected_categories and len(selected_categories) > 1:
            group_col = 'Categoria'

        if group_col:
            fig = px.box(
                filtered_data,
                x='DiaSemana',
                y='Unidades Vendidas',
                color=group_col,
                title=f"Distribuição por Dia da Semana por {group_col}",
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
        if (not selected_products or len(selected_products) < 2) and (
                not selected_categories or len(selected_categories) < 2):
            st.info("Selecione pelo menos 2 produtos ou categorias para comparação")
        else:
            # Agrupar por produto/categoria e data
            group_col = None
            if selected_products and len(selected_products) > 1:
                group_col = 'Produto'
            elif selected_categories and len(selected_categories) > 1:
                group_col = 'Categoria'

            comparison_data = filtered_data.groupby([group_col, pd.Grouper(key='Data', freq='W')])[
                'Unidades Vendidas'].sum().reset_index()

            fig = px.line(
                comparison_data,
                x='Data',
                y='Unidades Vendidas',
                color=group_col,
                title=f"Comparação Semanal de Vendas por {group_col}",
                labels={'Unidades Vendidas': 'Unidades Vendidas', 'Data': 'Data'}
            )

            fig.update_layout(
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Distribuição das Vendas")

        group_col = None
        if selected_products and len(selected_products) > 1:
            group_col = 'Produto'
        elif selected_categories and len(selected_categories) > 1:
            group_col = 'Categoria'

        if group_col:
            fig = px.histogram(
                filtered_data,
                x='Unidades Vendidas',
                color=group_col,
                marginal="box",
                title=f"Distribuição de Vendas por {group_col}",
                barmode="overlay"
            )
        else:
            fig = px.histogram(
                filtered_data,
                x='Unidades Vendidas',
                marginal="box",
                title="Distribuição de Vendas"
            )

        st.plotly_chart(fig, use_container_width=True)


def show_stats_tab(system: AdvancedDemandForecastSystem) -> None:
    """Exibe a aba de estatísticas aprimorada"""
    st.header("Estatísticas da Previsão")

    if not system.forecast_results:
        st.warning("Execute uma previsão na aba '🔮 Previsão' para ver estatísticas")
        return

    forecast_df = system.forecast_results['forecast']
    forecast_data = forecast_df['Unidades Previstas']

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Média Diária", f"{forecast_data.mean():.1f} unidades")
    col2.metric("Mínimo Diário", f"{forecast_data.min():.1f} unidades")
    col3.metric("Máximo Diário", f"{forecast_data.max():.1f} unidades")
    col4.metric("Total Previsto", f"{forecast_data.sum():.1f} unidades")

    # Mostrar métricas de erro se existirem
    if system.forecast_results.get('error_metrics'):
        st.divider()
        st.subheader("Métricas de Qualidade da Previsão")

        cols = st.columns(3)
        cols[0].metric("MAPE (Erro %)", f"{system.forecast_results['error_metrics']['MAPE']:.2f}%")
        cols[1].metric("RMSE", f"{system.forecast_results['error_metrics']['RMSE']:.2f}")
        cols[2].metric("Taxa de Acerto", f"{system.forecast_results['error_metrics']['Accuracy']:.2f}%")

    st.divider()

    # Análise por dia da semana
    forecast_df['Dia da Semana'] = forecast_df['Data'].dt.day_name()

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
            'Unidades Previstas': '{:.1f}',
            'Previsão Mínima': '{:.1f}',
            'Previsão Máxima': '{:.1f}'
        }),
        column_order=['Data', 'Dia da Semana', 'Unidades Previstas', 'Previsão Mínima', 'Previsão Máxima'],
        use_container_width=True,
        hide_index=True
    )


def show_export_tab(system: AdvancedDemandForecastSystem) -> None:
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
        backup_dates = [f"Backup {i + 1} ({len(backup['data'])} registros, {len(backup['forecast_history'])} previsões)"
                        for i, backup in enumerate(st.session_state.data_backups)]

        selected_backup = st.selectbox(
            "Selecione um backup para restaurar",
            options=backup_dates
        )

        if st.button("Restaurar Backup"):
            index = backup_dates.index(selected_backup)
            system.data = st.session_state.data_backups[index]['data'].copy()
            system.forecast_results = st.session_state.data_backups[index]['forecast_results'].copy() if \
            st.session_state.data_backups[index]['forecast_results'] else None
            system.forecast_history = st.session_state.data_backups[index]['forecast_history'].copy()
            system.save_data()
            st.success("Backup restaurado com sucesso!")
            st.rerun()
    else:
        st.info("Nenhum backup disponível")


def show_settings_tab(system: AdvancedDemandForecastSystem) -> None:
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
        Versão 4.0  
        Desenvolvido para gestão profissional de demanda  

        **Recursos principais:**  
        - Múltiplos modelos de previsão (ARIMA, SARIMA, Holt-Winters, Prophet, Random Forest, XGBoost)  
        - Suporte a múltiplos produtos e categorias  
        - Intervalos de confiança nas previsões  
        - Métricas de qualidade da previsão (MAPE, RMSE, Taxa de Acerto)  
        - Gráficos interativos  
        - Otimização automática de parâmetros  
        - Exportação para Excel e PDF  

        © 2023 - Todos os direitos reservados
    """)


if __name__ == "__main__":
    main()