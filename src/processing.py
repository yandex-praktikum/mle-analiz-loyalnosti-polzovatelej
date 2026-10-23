import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from phik import phik_matrix
from sqlalchemy import create_engine 
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine

class DataProcessor:

    def __init__(self, orders_path: str, rates_path: Optional[str] = None, db_config: Optional[Dict[str, Any]] = None):
        """
        :param db_config: можно передать вручную (для тестов), 
                          или оставить None — тогда прочитается из ENV.
        """
        if db_config is None:
            self.db_config = self._get_db_config_from_env()
        else:
            self.db_config = db_config

        self.engine = None
        self._initialize_engine()

        self.orders_path = Path(orders_path)
        self.rates_path = Path(rates_path) if rates_path else None

        self.df: Optional[pd.DataFrame] = None
        self.df_rates: Optional[pd.DataFrame] = None

    # Загрузка данных
    def load_data(self) -> "DataProcessor":
        print("\n--- Загрузка данных ---")
        self.load_orders()
        self.load_rates()
        self.convert_currency_to_rub()
        print(f"Всего записей в датафрейме: {self.df.shape[0]}")
        return self

    # Предобработка
    def clean_and_filter_data(self, col_negative: str, col_percentile: str, percentile: float) -> "DataProcessor":
        print(f"\n--- Обработка данных ---")
        self.remove_negative_revenue(col_negative)
        self.remove_duplicate_orders()
        self.filter_by_percentile(col_percentile, percentile)
        return self

    # Загрузка датафрейма с данными из базы или файла. Обработка типов данных
    def load_orders(self) -> "DataProcessor":
        """
        Загружает данные: сначала пытается из БД, при ошибке — из локального CSV.
        
        :param local_path: путь к локальному файлу, если БД недоступна.
        :return: DataFrame с данными.
        """
        dtype_spec = {
            "order_id": "int32",
            "event_id": "int32",
            "tickets_count": "int8",
            "service_name": "category",
            "device_type_canonical": "category",
            "region_name": "category",
            "city_name": "category",
            "event_type_main": "category",
        }

        query = self._build_query()

        # Попытка загрузки из БД
        if self.engine is not None:
            try:
                self.df = pd.read_sql_query(
                query,
                con=self.engine,
                parse_dates=["order_dt", "order_ts"],
                dtype=dtype_spec)
                print(f"Загрузка из базы прошла корректно")
                return self
            except Exception as e:
                print(f"База недоступна ({e}), читаем локальную копию")

        # Fallback на локальный файл
        if os.path.exists(self.orders_path):
            self.df = pd.read_csv(
            self.orders_path,
            parse_dates=["order_dt", "order_ts"],
            dtype=dtype_spec,
            on_bad_lines="skip",
        )
            print(f"Загрузка данных из файла {self.orders_path}")
            return self
        else:
            raise FileNotFoundError(
                f"Не удалось подключиться к БД и не найден локальный файл: {self.orders_path}"
            )

    # Загрузка файла с курсом валют. Обработка типов данных
    def load_rates(self) -> "DataProcessor":

        """Загружает CSV с курсом валют и сразу оптимизирует тип дата."""
        if not self.rates_path.exists():
            raise FileNotFoundError(f"Файл с курсом валют не найден: {self.rates_path}")

        self.df_rates = pd.read_csv(self.rates_path, parse_dates=["data"])
        return self

    # Объединение датафреймов и конвертация в рубли    
    def convert_currency_to_rub(self, output_col: str = "revenue_rub") -> "DataProcessor":
        
        # Проверка датафрейма df на существование:     
        if self.df is None:
            raise ValueError("Сначала вызовите load_orders()")

        df = self.df
        df_rates = self.df_rates

        # Объединим датафреймы 
        df_merged = df.merge(df_rates, left_on=['order_dt', 'currency_code'], right_on=['data', 'cdx'], how='left')

        # Переведем тенге в рубли 
        df_merged[output_col] = df_merged['revenue'] * df_merged['curs'] / df_merged['nominal']

        # Заполним поле revenue_rub значениями из поля revenue для рублевой валюты 
        mask = df_merged['currency_code'] == 'rub'   
        df_merged.loc[mask, output_col] = df_merged.loc[mask, 'revenue']

        # Округлим значение суммы до двух знаков после запятой
        df_merged[output_col] = df_merged[output_col].round(2)

        # Удалим лишние столбцы из датафрейма
        df_merged = df_merged.drop(columns=['data', 'nominal', 'curs', 'cdx'])

        self.df = df_merged
        return self

    # Предобработка данных. Обработка дубликатов в столбце order_id    
    def remove_duplicate_orders(self) -> "DataProcessor":
        df_drop = self.df

        # Выберем колонки по которым будем сравнивать заказы
        cols = ['user_id', 'order_ts', 'device_type_canonical', 'event_id', 'service_name', 'tickets_count', 'revenue_rub']

        # Подсчитаем количество и долю дублей
        duplicates = df_drop.duplicated(subset=cols).sum()

        # Удалим неявные дубли
        df = df_drop.drop_duplicates(subset=cols, keep='first').reset_index(drop=True)
        print(f"Удаление дубликатов заказов: {duplicates}")
        self.df = df

        return self

    # Предобработка данных. Удаление отрицательной выручки
    def remove_negative_revenue(self, col: str) -> "DataProcessor":
        df= self.df

        # Найдем количество записей с revenue_rub меньше нуля
        negative = df[df[col] < 0].shape[0]

        # Найдем долю записей с отрицательным revenue_rub
        negative_ratio = negative / df.shape[0] 

        # Оставим в датафрейме только записи с неотрицательной выручкой
        df_final = df[df[col] >= 0].reset_index(drop=True)

        # Выведем информацию о датафрейме
        print(f"Удаление записей с отрицательным {col}: {negative}")

        self.df = df_final
        return self
    
    # Предобработка данных. Фильтр в указанной колонке по 99 перцентилю
    def filter_by_percentile(self, col: str, upper_percentile: float = 99.0) -> "DataProcessor":
        df = self.df
        
        # Найдем 99 перцентиль
        q_high = self.df[col].quantile(upper_percentile / 100.0)

        # Отфильтруем данные по 99 перцентилю
        df_99 = df.loc[df[col] <= q_high]
        print(f"Фильтрация записей полю {col} по {upper_percentile} перцентилю (значение {q_high})")
        print(f"Осталось записей после фильтрации: {df_99.shape[0]}")
        self.df = df_99
        return self

    # Предобработка данных. Проверка значений категориальных столбцов на дубликаты
    def check_categorical_consistency(self, cols: list) -> "DataProcessor":
        df = self.df
        for col in cols:
            # Выводим уникальные значения в столбце 
            count_items = df[col].nunique()   
            # Приводим к нижнему регистру и считаем количество уникальных значений
            count_lower_case = df[col].str.lower().nunique()
            if count_items != count_lower_case:
                print(f"Дубликаты в столбце {col}")
        print(f"Дубликатов в категориальных столбцах нет") 
        self.df = df
        return self

    # Создание профиля пользователя
    def create_user_profile(self) -> "DataProcessor":
        print(f"\n--- Создание профиля пользователя ---")

        df = self.df
        # Отсортируем данные по времени совершения заказа
        sorted_df = df.sort_values(by='order_ts', ascending=True)

        # Создаем профиль пользователя
        profile = (sorted_df.groupby('user_id')
                .agg(first_order_date=('order_ts', 'min'),
                        last_order_date=('order_ts', 'max'),
                        first_order_device=('device_type_canonical', 'first'),
                        first_order_region=('region_name', 'first'),
                        first_order_event=('event_type_main', 'first'),
                        count_orders_per_user=('user_id', 'size'),
                        mean_order_revenue_per_user=('revenue_rub', 'mean'),
                        mean_time_between_orders=('days_since_prev', 'mean'),
                        sum_revenue_per_user=('revenue_rub', 'sum'),
                        is_two=('user_id', (lambda x: len(x) > 1))
                    ).reset_index())

        # Округляем до двух знаков после запятой
        profile['mean_order_revenue_per_user'] = profile['mean_order_revenue_per_user'].round(2)
        profile['mean_time_between_orders'] = profile['mean_time_between_orders'].round(2)
        profile['sum_revenue_per_user'] = profile['sum_revenue_per_user'].round(2)
        self.df = profile
        return self

    # Общее число пользователей в выборке
    def count_unique_users(self) -> int:
        df = self.df
        uniq_users_count = df['user_id'].nunique() 
        return uniq_users_count
    
    # Средняя выручка с одного заказа (общее по всем заказам)
    def calculate_avg_order_value(self) -> float:
        df = self.df
        mean_revenue = df['sum_revenue_per_user'].sum() / df['count_orders_per_user'].sum()
        return mean_revenue
    
    # Доля пользователей по всему датафрейму, сделавших заказ повторно
    def compute_repeat_purchase_rate(self) -> float:
        df = self.df
        # Доля пользователей, сделавших заказ повторно
        mean_ratio = round(df['is_two'].mean(), 4)
        return mean_ratio

    # Общее количество заказов в датафрейме
    def count_total_orders(self) -> int:
        df = self.df
        count_orders = df['count_orders_per_user'].sum()
        return count_orders

    def report_profile_metrics(self) -> "DataProcessor":

        count_orders = self.count_total_orders()
        print(f"Общее количество заказов в датафрейме: {count_orders}")

        mean_ratio = self.compute_repeat_purchase_rate()
        print(f"Доля пользователей, сделавших заказ повторно: {mean_ratio}")

        uniq_users_count = self.count_unique_users()
        print(f"Число уникальных пользователей в выборке: {uniq_users_count}")

        mean_revenue = self.calculate_avg_order_value()
        print(f"Средняя выручка с одного заказа (общее по всем заказам): {mean_revenue.round(2)} руб.")

        return self

    # Расчет количества и доли пользователей, совершивших заказ повторно
    def analyze_retention_by_segments(self, cols: list) -> "DataProcessor":
        
        df = self.df
        mean_ratio = self.compute_repeat_purchase_rate()
        for col in cols:
            print(f"\n--- Расчет количества и доли пользователей по {col}, совершивших заказ повторно ---")
            count_series = df[col].value_counts() 
            ratio_series = round(count_series / len(df), 4)

            # Расчет количества и доли пользователей, совершивших заказ повторно
            stats = (df.groupby(col).agg(
                share_two=('is_two', 'mean'),      # доля пользователей с 2+ заказами
                total_users=('user_id', 'count')       # общее количество пользователей
            ).reset_index()
            .assign(share_two=lambda x: x['share_two'].round(4))
            .sort_values(by='total_users', ascending = False)
            )
            
            if len(stats) >= 10:
                stats = stats.head(10).reset_index()

            stats = stats.sort_values(by='share_two', ascending = False).reset_index()

            for i in range(len(count_series)):
                print(f"На {i+1} месте: {count_series.index[i]} - {count_series.iloc[i]} с долей {ratio_series.iloc[i]}")
                if i == 9:
                    break

            print(f"\nДоля повторных заказов выше среднего для: ")
            for i in range(len(stats)): 
                if stats['share_two'][i] > mean_ratio:
                    print(f"  - {stats[col][i]} - {stats['share_two'][i]}")
                if i == 6:
                    break
        self.df = df        
        return self

    # Гипотеза 1
    def test_hypothesis_event_repeat_rate(self, col: str,  more_ratio: str, min_ratio: str) -> "DataProcessor":
        print(f"\n--- Гипотеза 1 --- ")
        print(f"Пользователи, которые совершили первый заказ на спортивные мероприятия, совершают повторный заказ чаще, чем пользователи, оформившие свой первый заказ на концерты.")
        df = self.df
        # Проводим подсчет общего количества пользователей в разрезе мероприятий и доли пользователей, совершивших заказ повторно
        stats = df.groupby(col).agg(
            share_two=('is_two', 'mean'),      # доля пользователей с 2+ заказами
            total_users=('user_id', 'count')       # общее количество пользователей
        ).reset_index().sort_values(by='total_users', ascending = False)

        share_more = round(stats.loc[stats[col] == more_ratio, 'share_two'].values[0], 4)
        share_min = round(stats.loc[df[col] == min_ratio, 'share_two'].values[0], 4)
        if share_more > share_min:
            print(f"Гипотеза 1 верна: доля {more_ratio} ({share_more}) выше доли {min_ratio} ({share_min})")
        else: 
            print(f"Гипотеза 1 неверна: доля {min_ratio} ({share_min}) выше доли {more_ratio} ({share_more})")

        self.df = df        
        return self

    # Гипотеза 2
    def test_hypothesis_region_activity_vs_retention(self, col: str) -> "DataProcessor":
        print(f"\n--- Гипотеза 2 --- ")
        print(f"В регионах, где больше всего пользователей посещают мероприятия, выше доля повторных заказов, чем в менее активных регионах.")
        df = self.df

        # Расчет количества и доли пользователей, совершивших заказ повторно
        stats = df.groupby(col).agg(
            share_two=('is_two', 'mean'),      # доля пользователей с 2+ заказами
            total_users=('user_id', 'count')       # общее количество пользователей
        ).reset_index().sort_values(by='total_users', ascending = False).head(10)
         
        # Топ‑3 по количеству пользователей
        top_by_users = stats.sort_values(by='total_users', ascending=False).head(3)
        print("Топ‑3 регионов по количеству пользователей:")
        print(top_by_users[[col, 'total_users', 'share_two']].to_string(index=False))

        # Топ‑3 по доле повторных заказов
        top_by_share = stats.sort_values(by='share_two', ascending=False).head(3)
        print("\nТоп‑3 регионов по доле повторных заказов:")
        print(top_by_share[[col, 'share_two', 'total_users']].to_string(index=False))

        # Сравнение топ‑3 по пользователям: смотрим, сохраняется ли тренд «больше пользователей → выше доля»
        regions = top_by_users[col].tolist()
        shares = top_by_users['share_two'].tolist()

        # Проверка монотонности: доля должна убывать или хотя бы не расти при переходе от 1→2→3
        is_monotonic_decreasing = all(shares[i] >= shares[i + 1] for i in range(len(shares) - 1))

        if is_monotonic_decreasing:
            print("\nВывод: в топ‑3 регионах доля повторных заказов убывает по мере снижения числа пользователей. "
                "Это поддерживает гипотезу: более активные регионы имеют более высокую долю повторных заказов.")
        else:
            # Найдём, где тренд ломается
            break_idx = next(i for i in range(len(shares) - 1) if shares[i] < shares[i + 1])
            print("Гипотеза 2 неверна: не во всех более активных регионах доля повторных выше.")

        self.df = df
        return self

    # Корреляционный анализ
    def run_phik_correlation_analysis( self,
                            feature_cols: Optional[List[str]] = None,
                            interval_cols: Optional[List[str]] = None,
                            target_col: str = "count_orders_per_user") -> "DataProcessor":
        """
        Корреляционный анализ через phi_k:
        - сначала по исходной целевой переменной,
        - затем по сегментированной версии этой переменной.
        
        Параметры:
        feature_cols — список колонок для анализа (если None — берутся по умолчанию).
        interval_cols — интервальные колонки для phik_matrix (если None — по умолчанию).
        target_col — целевая колонка, которую будем сегментировать.
        """

        print(f"\n--- Корреляционный анализ по переменной interval_cols_seg ---")
        df_phik = self.df.copy()

        # Колонки по умолчанию
        if feature_cols is None:
            feature_cols = [
                "first_order_device",
                "first_order_region",
                "first_order_event",
                "count_orders_per_user",
                "mean_order_revenue_per_user", 
                "mean_time_between_orders"
            ]

        # Числовые колонки
        if interval_cols is None:
            interval_cols = [
                "count_orders_per_user",
                "mean_order_revenue_per_user",
                "mean_time_between_orders"
            ]

        # Сегментация целевой переменной 
        segment_bins = [0, 1, 2, 4, float("inf")]
        segment_labels = ["1 заказ", "2 заказа", "3–4 заказа", "5+ заказов"]

        segment_col = f"{target_col}_segment"
        df_phik[segment_col] = pd.cut(
            df_phik[target_col],
            bins=segment_bins,
            labels=segment_labels,
            right=True,
            include_lowest=True,
        )

        print(df_phik[segment_col].value_counts())

        # Заменяем target_col на segment_col
        cols_with_segment = [c if c != target_col else segment_col for c in feature_cols]

        # Обновляем interval_cols: убираем из него target_col (теперь это категория)
        interval_cols_seg = [c for c in interval_cols if c != target_col]

        # Корреляция по переменной interval_cols_seg
        corr_seg = df_phik[cols_with_segment].phik_matrix(interval_cols=interval_cols_seg)

        mask_seg = corr_seg.index != segment_col
        segment_corr = (
            corr_seg.loc[mask_seg, [segment_col]]
            .sort_values(by=segment_col, ascending=False)
        )

        # Визуализация
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            segment_corr,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Коэффициент phik"}
        )
        plt.title(f"Связь сегмента по числу заказов с другими признаками")
        plt.xlabel("Признак")
        plt.tight_layout()
        plt.show()
        return self

    # Чтение из конфига
    @staticmethod
    def _get_db_config_from_env() -> Optional[Dict[str, Any]]:
        """
        Читает параметры подключения из переменных окружения.
        Возвращает словарь или None, если хотя бы один ключ отсутствует.
        """
        load_dotenv()
        user = os.getenv("DB_USER")
        pwd = os.getenv("DB_PWD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db = os.getenv("DB_NAME")

        # Если чего-то не хватает — не создаём конфиг (fallback на локальный файл)
        if not all([user, pwd, host, port, db]):
            print("Не хватает одной или нескольких переменных окружения — возвращаем None")
            return None

        return {
            "user": user,
            "pwd": pwd,
            "host": host,
            "port": int(port),
            "db": db,
        }

    # Запрос к БД
    @staticmethod
    def _build_query() -> str:
        """Возвращает SQL-запрос для загрузки данных."""
        return """
        SELECT
            p.user_id,
            p.device_type_canonical, 
            p.order_id,
            p.created_dt_msk AS order_dt, 
            p.created_ts_msk AS order_ts,
            p.currency_code, 
            p.revenue, 
            p.tickets_count,
            EXTRACT(DAY FROM (created_dt_msk - LAG(created_dt_msk) OVER (
                PARTITION BY user_id
                ORDER BY created_dt_msk
            ))) AS days_since_prev,
            p.event_id,
            e.event_name_code AS event_name,
            e.event_type_main, 
            p.service_name, 
            r.region_name,
            c.city_name  
        FROM afisha.purchases p
        JOIN afisha.events e ON p.event_id = e.event_id
        JOIN afisha.city c ON e.city_id = c.city_id
        JOIN afisha.regions r ON c.region_id = r.region_id
        WHERE (device_type_canonical IN ('mobile', 'desktop'))
          AND (e.event_type_main != 'фильм')
        ORDER BY user_id;
        """

    # Создание движка SQLAlchemy
    def _initialize_engine(self) -> None:
        """Создаёт движок SQLAlchemy, если есть конфиг БД."""
        if not self.db_config:
            return

        connection_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['pwd']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['db']}"
        )
        self.engine = create_engine(connection_string)

