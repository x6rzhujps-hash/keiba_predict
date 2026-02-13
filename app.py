import streamlit as st
import lightgbm as lgb
import pandas as pd
from pathlib import Path
import os

PASSWORD = os.environ.get("APP_PASSWORD")

pw = st.text_input("パスワードを入力", type="password")

if pw != PASSWORD:
    st.stop()

st.success("ログイン成功")

# =========================
# あなたの既存関数を import
# =========================
from scraping_features import (
    extract_race_id,
    get_race_info,
    get_shutuba_data,
    get_last_races,
    process_race_info_features,
    process_shutuba_features,
    build_last_race_features,
    attach_race_info,
    attach_class_diff,
)

# =========================
# モデルロード
# =========================
@st.cache_resource
def load_model_ability():
    MODEL_PATH = "models/lightgbm_v5/lgbm_ability_v2.txt"
    return lgb.Booster(model_file=str(MODEL_PATH))

def load_model_pace():
    MODEL_PATH = "models/lightgbm_v5/lgbm_pace_v2.txt"
    return lgb.Booster(model_file=str(MODEL_PATH))

model_ability = load_model_ability()
model_pace = load_model_pace()

FEATURE_COLS_ABILITY = [
    'sex', 'age', 'jockey_weight', 'body_weight', 'body_weight_diff', 'n_valid_races', 'has_dummy_race', 'rank_min', 'time_diff_mean', 'last3f_mean', 'distance_mean', 'corner_mean', 'body_weight_mean', 'body_weight_diff_mean', 'jockey_weight_mean', 'past_class_mean', 'past_class_max', 'past_class_last', 'class_diff_mean', 'class_diff_last', 'past_same_distance_rate', 'past_distance_diff_mean', 'past_same_surface_rate', 'past_surface_rank_mean', 'past_same_course_rate', 'past_same_condition_rate', 'rank_std', 'time_diff_std', 'top3_rate', 'bad_finish_rate', 'rank_trend', 'time_diff_trend', 'rank_improve_from_mean', 'time_diff_improve_from_mean', 'corner_rank_mean', 'first_corner_mean', 'last_corner_mean', 'corner_gain_mean', 'corner_gain_rate_mean', 'is_mover_rate', 'days_since_last_race', 'interval_mean', 'interval_min', 'short_interval_rate', 'long_rest_flag', 'rank_mean_weighted_by_class', 'rank_last_weighted_by_class', 'time_diff_mean_weighted_by_class', 'top3_rate_weighted_by_class', 'race_class_encoded', 'race_course_encoded', 'surface_encoded', 'condition_encoded', 'distance'
]

FEATURE_COLS_PACE = [
    'frame', 'sex', 'age', 'jockey_weight', 'body_weight', 'body_weight_diff', 'n_valid_races', 
    'has_dummy_race', 'corner_mean', 'past_same_distance_rate', 'past_distance_diff_mean', 
    'past_same_surface_rate', 'past_surface_rank_mean', 'past_same_course_rate', 
    'past_same_condition_rate', 'corner_rank_mean', 'n_horses_ahead_mean', 'n_horses_behind_mean', 
    'front_rate', 'front_density_mean', 'front_margin_mean', 'first_corner_mean', 'last_corner_mean', 
    'corner_gain_mean', 'corner_gain_rate_mean', 'is_mover_rate', 'early_position_rate', 
    'early_x_shortening', 'mover_x_extension', 'early_x_small_field', 'mover_x_large_field', 
    'pace_pressure_index', 'race_class_encoded', 'race_course_encoded', 'surface_encoded', 
    'condition_encoded', 'distance', 'number_of_horse'
]

# =========================
# UI
# =========================
st.title("🏇 競馬AI LIGHTGBM V5：3着以内予測")

race_input = st.text_input(
    "レースID または 出馬表URLを入力してください",
    placeholder="例: 202606010611 または netkeiba のURL"
)

if st.button("予想する"):
    race_id = extract_race_id(race_input)
    if race_id is None:
        st.warning("有効なレースIDまたはURLを入力してください")
        st.stop()

    with st.spinner("データ取得・予測中..."):
        # ===== データ取得 =====
        race_info_df = get_race_info(race_id)
        race_info_df_copy = race_info_df.copy()
        shutuba_df = get_shutuba_data(race_id)
        shutuba_df_copy = shutuba_df.copy()
        last_races_df = get_last_races(race_id)

        # ===== 特徴量 =====
        race_info_df = process_race_info_features(race_info_df)
        shutuba_feat = process_shutuba_features(shutuba_df)
        last_feat = build_last_race_features(last_races_df, race_info_df)

        df = shutuba_feat.merge(last_feat, on="horse_no", how="left")
        df = attach_race_info(df, race_info_df)
        df = attach_class_diff(df)
        df["race_id"] = race_id

        X_ABILITY = df[FEATURE_COLS_ABILITY]
        X_ABILITY = X_ABILITY.fillna(0)
        assert list(X_ABILITY.columns) == FEATURE_COLS_ABILITY

        X_PACE = df[FEATURE_COLS_PACE]
        X_PACE = X_PACE.fillna(0)
        assert list(X_PACE.columns) == FEATURE_COLS_PACE

        # ===== 予測 =====
        df = df.reset_index(drop=True)

        df["pred_top3_prob_ability"] = model_ability.predict(X_ABILITY)
        df["pred_top3_prob_pace"] = model_pace.predict(X_PACE)

        # total予測（最終スコア）
        df["pred_top3_prob"] = df["pred_top3_prob_ability"] + df["pred_top3_prob_pace"]

        # =========================
        # ability top5
        # =========================
        result_ability = (
            df[["horse_no", "pred_top3_prob_ability"]]
            .merge(
                shutuba_df_copy[["horse_no", "horse_name"]],
                on="horse_no",
                how="left"
            )
            .sort_values("pred_top3_prob_ability", ascending=False)
            .head(5)
            .copy()
        )

        result_ability["pred_top3_prob_ability"] = result_ability["pred_top3_prob_ability"].round(3)

        # =========================
        # total top5（今までの予測）
        # =========================
        result_total = (
            df[["horse_no", "pred_top3_prob"]]
            .merge(
                shutuba_df_copy[["horse_no", "horse_name"]],
                on="horse_no",
                how="left"
            )
            .sort_values("pred_top3_prob", ascending=False)
            .head(5)
            .copy()
        )

        result_total["pred_top3_prob"] = result_total["pred_top3_prob"].round(3)

    # =========================
    # 出力
    # =========================

    race_info = race_info_df_copy.iloc[0]
    st.subheader("📋 レース情報")
    st.write(
        f"**{race_info['race_name']}**  "
        f"（{race_info['race_course']}・{race_info['race_class']}・{race_info['surface']}・{int(race_info['distance'])}・{int(race_info['number_of_horse'])}頭）"
    )

    # ===== ability top5 =====
    st.subheader("🔥 馬の能力（abilityモデル）TOP5")

    result_ability = result_ability.reset_index(drop=True)

    for i, row in result_ability.iterrows():
        st.write(
            f"**{i+1}位**："
            f"{int(row.horse_no)}番 "
            f"{row.horse_name} "
            f"(能力スコア={row.pred_top3_prob_ability})"
        )

    # ===== total top5 =====
    st.subheader("🔮 最終予測（ability + pace）TOP5")

    result_total = result_total.reset_index(drop=True)

    for i, row in result_total.iterrows():
        st.write(
            f"**{i+1}位予想**："
            f"{int(row.horse_no)}番 "
            f"{row.horse_name} "
            f"(3着内確率={row.pred_top3_prob})"

        )


