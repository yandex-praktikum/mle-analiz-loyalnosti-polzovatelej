import pandas as pd
from src.processing import DataProcessor

processor = DataProcessor("data/raw_data.csv", "data/final_tickets_tenge_df.csv")

df = (processor
    .load_data()
    .clean_and_filter_data("revenue_rub", "revenue_rub", 99)
    .check_categorical_consistency(["device_type_canonical", "event_type_main", "service_name", "region_name", "city_name"])
    .create_user_profile()
    .filter_by_percentile("count_orders_per_user",99)
    .report_profile_metrics()
    .analyze_retention_by_segments(["first_order_event", "first_order_device", "first_order_region"])
    .test_hypothesis_event_repeat_rate("first_order_event", "спорт", "выставки")
    .test_hypothesis_region_activity_vs_retention("first_order_region")
    .run_phik_correlation_analysis() 
    )          


