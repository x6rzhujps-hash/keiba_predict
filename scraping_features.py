import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from io import StringIO
import re
from pathlib import Path


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


JRA_COURSES = [
    "札幌","函館","福島","新潟","東京","中山",
    "中京","京都","阪神","小倉"
]

RACE_CLASS_MAP = {
    "GI": 9,
    "JpnI": 9,
    "JGI": 9,
    "GII": 8,
    "JpnII": 8,
    "JGII": 8,
    "GIII": 7,
    "JpnIII": 7,
    "JGIII": 7,
    "L": 6,
    "OP": 5,
    "3勝": 4,
    "2勝": 3,
    "1勝": 2,
    "新馬": 1,
    "未勝利": 0,
    "J未勝利": 0,
}

KANJI_RE = re.compile(r'[一-龥]')

COURSE_MAP = {course: i for i, course in enumerate(JRA_COURSES)}
OTHER_COURSE_ID = 10

CONDITION_MAP = {
    "良": 3,
    "稍": 2,
    "重": 1,
    "不良": 0,
}
session = requests.Session()
session.headers.update(HEADERS)

def extract_race_id(input_text: str) -> str | None:
    """
    race_id そのもの or netkeiba のURL から race_id を抽出
    """
    input_text = input_text.strip()

    # すでに race_id 形式
    if re.fullmatch(r"\d{12}", input_text):
        return input_text

    # URL から抽出
    m = re.search(r"race_id=(\d{12})", input_text)
    if m:
        return m.group(1)

    return None

def fetch_with_retry(
    session,
    url,
    headers=None,
    max_retry=3,
    timeout=(5, 10),
):
    last_error = None

    for i in range(max_retry):
        try:
            res = session.get(url, headers=headers, timeout=timeout)
            res.raise_for_status()
            return res

        except Exception as e:
            last_error = e
            if i < max_retry - 1:
                time.sleep(2)

    raise last_error

def get_race_info(race_id: str):

    url = (
        "https://race.netkeiba.com/race/shutuba.html"
        f"?race_id={race_id}&rf=shutuba_submenu"
    )
    # res = session.get(url, timeout=30)
    res = fetch_with_retry(
        session,
        url,
        headers=HEADERS,
    )
    res.encoding = "EUC-JP"

    soup = BeautifulSoup(res.text, "lxml")

    # ======================
    # レース情報ブロック
    # ======================
    race_block = soup.find("div", class_="RaceList_Item02")

    if race_block is None:
        raise ValueError("RaceList_Item02 が見つかりません（HTML構造変更の可能性）")

    # ======================
    # レース名
    # ======================
    race_name = race_block.find("h1", class_="RaceName").get_text(strip=True)

    # ======================
    # グレード（GI, GII, GIII）
    # ======================
    race_class = None
    grade_map = {
        "Icon_GradeType1": "GI",
        "Icon_GradeType2": "GII",
        "Icon_GradeType3": "GIII",
        "Icon_GradeType5": "OP",
        "Icon_GradeType10": "JGI",
        "Icon_GradeType11": "JGII",
        "Icon_GradeType12": "JGIII",
        "Icon_GradeType15": "L",
        "Icon_GradeType16": "3勝",
        "Icon_GradeType17": "2勝",
        "Icon_GradeType18": "1勝",
    }

    grade_span = race_block.find("span", class_=re.compile("Icon_GradeType"))
    if grade_span:
        for cls in grade_span.get("class", []):
            if cls in grade_map:
                race_class = grade_map[cls]
                break

    # --- ② アイコンがない場合：レース名から推定 ---
    if race_class is None:
        if "障害未勝利" in race_name:
            race_class = "J未勝利"
        elif "未勝利" in race_name:
            race_class = "未勝利"
        elif "新馬" in race_name:
            race_class = "新馬"
        elif "1勝クラス" in race_name:
            race_class = "1勝"
        elif "2勝クラス" in race_name:
            race_class = "2勝"
        elif "3勝クラス" in race_name:
            race_class = "3勝"
        else:
            race_class = "未勝利"  # 最後の逃げ道

    # ======================
    # RaceData01（芝/ダ・距離・馬場）
    # ======================
    race_data01 = race_block.find("div", class_="RaceData01").get_text(" ", strip=True)

    # 芝 or ダート + 距離
    surface = None
    distance = None
    m = re.search(r'(芝|ダ)(\d+)m', race_data01)
    if m:
        if m.group(1) == 'ダ':
            surface = 'ダート'
        else:
            surface = m.group(1)
        distance = float(m.group(2))
    else:
        # 障害レース
        m_obs = re.search(r'障(\d+)m', race_data01)
        if m_obs:
            surface = '障害'
            distance = float(m_obs.group(1))

    # 馬場状態
    ground = None
    m = re.search(r'馬場[:：]\s*(良|稍|重|不良)', race_data01)
    if m:
        ground = m.group(1)

    # ======================
    # RaceData02（競馬場・頭数）
    # ======================
    race_data02 = race_block.find("div", class_="RaceData02").get_text(" ", strip=True)

    # 競馬場（国内10場 or 海外）
    jra_courses = ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]
    race_course = "海外"
    for c in jra_courses:
        if c in race_data02:
            race_course = c
            break

    # 頭数
    num_horses = None
    m = re.search(r'(\d+)頭', race_data02)
    if m:
        num_horses = int(m.group(1))

    # ======================
    # 結果表示
    # ======================
    race_info = {
        "race_name": race_name,
        "race_course": race_course,
        "race_class": race_class,
        "surface": surface,
        "distance": distance,
        "condition": ground,
        "number_of_horse": num_horses,
    }

    df_race_info = pd.DataFrame([race_info])

    return df_race_info

def parse_body_weight(text):
    """
    '516(+4)' → (516, 4)
    '---'     → (None, None)
    """
    if pd.isna(text):
        return None, None

    match = re.search(r"(\d+)\(([-+]?\d+)\)", str(text))
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return None, None

def get_shutuba_data(race_id: str):

    url = (
        "https://race.netkeiba.com/race/shutuba.html"
        f"?race_id={race_id}&rf=shutuba_submenu"
    )
    # res = session.get(url, timeout=30)
    res = fetch_with_retry(
        session,
        url,
        headers=HEADERS,
    )
    res.encoding = "EUC-JP"

    html = StringIO(res.text)
    tables = pd.read_html(html)
    df = tables[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "枠": "frame",
        "馬 番": "horse_no",
        "馬番": "horse_no",
        "馬名": "horse_name",
        "性齢": "sex_age",
        "斤量": "jockey_weight",
        "騎手": "jockey",
        "厩舎": "trainer",
        "馬体重 (増減)": "body_weight_raw",
    })

    use_cols = [
        "frame",
        "horse_no",
        "horse_name",
        "sex_age",
        "jockey_weight",
        "jockey",
        "trainer",
        "body_weight_raw",
    ]

    df = df[use_cols]

    df[["body_weight", "body_weight_diff"]] = (
        df["body_weight_raw"]
        .apply(parse_body_weight)
        .apply(pd.Series)
    )

    df = df.drop(columns=["body_weight_raw"])

    return df

def parse_race_text(race_text):
    if pd.isna(race_text):
        return {
            "race_date": None,
            "race_course": None,
            "rank": None,
            "race_class": None,
            "surface": None,
            "distance": None,
            "time": None,
            "condition": None,
            "num_horses": None,
            "horse_number": None,
            "corner": None,
            "last3f": None,
            "body_weight": None,
            "body_weight_diff": None,
            "time_difference": None,
            "jockey": None,
            "jockey_weight": None,
            "trainer": None,
        }

    text = race_text.replace("\u3000", " ").replace("\xa0", " ").strip()

    # ----------------------------
    # 日付
    # ----------------------------
    date_match = re.search(r"(\d{4}\.\d{2}\.\d{2})", text)
    race_date = date_match.group(1) if date_match else None

    # ----------------------------
    # 競馬場 + 着順
    # ----------------------------
    course_rank_match = re.search(r"\d{4}\.\d{2}\.\d{2}\s+([^\s]+)\s+(\d+)", text)
    race_course = None
    rank = None
    if course_rank_match:
        course_raw = course_rank_match.group(1)
        rank = int(course_rank_match.group(2))
        if re.search(r"[一-龥]", course_raw):
            race_course = course_raw
        else:
            race_course = "海外"

    # ----------------------------
    # race_class
    # ----------------------------
    race_class = None
    for g in ["JGI", "JGII", "JGIII", "GI", "GII", "GIII", "JpnI", "JpnII", "JpnIII"]:
        if g in text:
            race_class = g
            break

    if race_class is None:
        if "リステッド" in text or " L " in text:
            race_class = "L"
        elif "オープン" in text or "OP" in text:
            race_class = "OP"
        elif "3勝" in text:
            race_class = "3勝"
        elif "2勝" in text:
            race_class = "2勝"
        elif "1勝" in text:
            race_class = "1勝"
        elif "新馬" in text:
            race_class = "新馬"
        elif "障害未勝利" in text:
            race_class = "J未勝利"
        elif "未勝利" in text:
            race_class = "未勝利"
        else :
            race_class = "未勝利"

    # ----------------------------
    # 芝 / ダート
    # ----------------------------
    surface = "芝" if "芝" in text else "ダート" if "ダ" in text else None

    # ----------------------------
    # 距離
    # ----------------------------
    dist_match = re.search(r"(芝|ダ)(\d+)", text)
    distance = float(dist_match.group(2)) if dist_match else None

    # ----------------------------
    # タイム
    # ----------------------------
    time_match = re.search(r"\s(\d+:\d+\.\d)\s", text)
    time = time_match.group(1) if time_match else None

    # ----------------------------
    # 馬場
    # ----------------------------
    condition_match = re.search(r"(良|稍|重|不)", text)
    condition = condition_match.group(1) if condition_match else None
    if condition == "不":
        condition = "不良"

    # ----------------------------
    # 頭数
    # ----------------------------
    num_match = re.search(r"(\d+)頭", text)
    num_horses = int(num_match.group(1)) if num_match else None

    # ----------------------------
    # 馬番
    # ----------------------------
    horse_no_match = re.search(r"(\d+)番", text)
    horse_number = int(horse_no_match.group(1)) if horse_no_match else None

    # ----------------------------
    # 通過順
    # ----------------------------
    corner_match = re.search(r"(\d+(?:-\d+)+)", text)
    corner = corner_match.group(1) if corner_match else None

    # ----------------------------
    # 上がり3F
    # ----------------------------
    last3f_match = re.search(r"\((\d+\.\d)\)", text)
    last3f = float(last3f_match.group(1)) if last3f_match else None
    if last3f == 0.0:
        last3f = np.nan

    # ----------------------------
    # 馬体重
    # ----------------------------
    bw_match = re.search(r"(\d+)\(([-+]?\d+)\)", text)
    body_weight = float(bw_match.group(1)) if bw_match else None
    weight_diff = float(bw_match.group(2)) if bw_match else None

    # ----------------------------
    # 着差
    # ----------------------------
    diff_match = re.search(r"\(([-+]?\d+\.\d)\)$", text)
    time_difference = float(diff_match.group(1)) if diff_match else None
    if time_difference == 0.0:
        time_difference = np.nan

    # ----------------------------
    # 騎手 + 斤量
    # ----------------------------
    jw_match = re.search(r"\d+人\s+([^\s]+)\s+(\d+\.\d)", text)
    jockey = jw_match.group(1) if jw_match else None
    jockey_weight = float(jw_match.group(2)) if jw_match else None

    return {
        "race_date": race_date,
        "race_course": race_course,
        "rank": rank,
        "race_class": race_class,
        "surface": surface,
        "distance": distance,
        "time": time,
        "condition": condition,
        "num_horses": num_horses,
        "horse_number": horse_number,
        "corner": corner,
        "last3f": last3f,
        "body_weight": body_weight,
        "body_weight_diff": weight_diff,
        "time_difference": time_difference,
        "jockey": jockey,
        "jockey_weight": jockey_weight,
        "trainer": None,  # ← 馬柱DFから後で結合
    }

def get_last_races(race_id: str):
    url = (
        "https://race.netkeiba.com/race/shutuba_past.html"
        f"?race_id={race_id}&rf=shutuba_submenu"
    )
    # res = session.get(url, timeout=30)
    res = fetch_with_retry(
        session,
        url,
        headers=HEADERS,
    )
    res.encoding = "EUC-JP"

    html = StringIO(res.text)
    tables = pd.read_html(html)
    df = tables[0]

    df = df.rename(columns={
        "枠": "frame",
        "馬番": "horse_no",
        "印": "mark",
        "馬名": "horse_name",
        "騎手 斤量": "jockey_weight",
        "前走": "last_race",
        "2走": "two_race",
        "3走": "three_race",
        "4走": "four_race",
        "5走": "five_race",
    })

    past_cols = ["last_race", "two_race", "three_race", "four_race", "five_race"]

    past_df = df.melt(
        id_vars=["frame", "horse_no", "horse_name"],
        value_vars=past_cols,
        var_name="race_index",
        value_name="race_text"
    )

    parsed = past_df["race_text"].apply(parse_race_text)
    parsed_df = pd.DataFrame(list(parsed))

    past_df = pd.concat([past_df, parsed_df], axis=1)
    return past_df

def load_race_info_with_fix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if df.loc[0, "condition"] is None:
        df = df.copy()
        df.loc[0, "condition"] = "稍"

    return df

def encode_race_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["race_class_encoded"] = df["race_class"].map(RACE_CLASS_MAP)

    # 未定義クラスがあれば検知
    unknown = df[df["race_class_encoded"].isna()]["race_class"].unique()
    if len(unknown) > 0:
        print("[WARN] Unknown race_class:", unknown)

    return df

def encode_race_course(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["race_course_encoded"] = (
        df["race_course"]
        .map(COURSE_MAP)
        .fillna(OTHER_COURSE_ID)
        .astype(int)
    )

    return df

def encode_surface(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["surface_encoded"] = df["surface"].map({
        "芝": 1,
        "ダート": 0,
        "障害": 2
    })

    return df

def encode_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["condition_encoded"] = df["condition"].map(CONDITION_MAP)
    return df

def process_race_info_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = encode_race_class(df)
    df = encode_race_course(df)
    df = encode_surface(df)
    df = encode_condition(df)

    return df[
        [
            "race_class_encoded",
            "race_course_encoded",
            "surface_encoded",
            "condition_encoded",
            "distance",
            "number_of_horse",
        ]
    ]

def split_sex_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sex_map = {
        "牡": 1,
        "牝": 0,
        "セ": 2,
    }

    df["sex"] = df["sex_age"].str[0].map(sex_map)
    df["age"] = df["sex_age"].str[1:].astype(int)

    return df

def process_shutuba_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = split_sex_age(df)

    return df[
        [
            "frame",
            "horse_no",
            "sex",
            "age",
            "jockey_weight",
            "body_weight",
            "body_weight_diff",
        ]
    ]

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_corners(x):
    if pd.isna(x):
        return []
    return [
        int(c) for c in str(x).split("-")
        if c.isdigit()
    ]

def preprocess_last_races(df):
    df = df.copy()

    # 数値化（既存）
    numeric_cols = [
        "rank",
        "time",
        "time_difference",
        "last3f",
        "distance",
        "body_weight",
        "body_weight_diff",
        "jockey_weight",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # corner → 最終コーナー（既存）
    df["last_corner"] = (
        df["corner"]
        .astype(str)
        .str.split("-")
        .str[-1]
    )
    df["last_corner"] = safe_numeric(df["last_corner"])

    # ===== ここからフェーズ2用の追加 =====

    # race_class → 数値化
    df["race_class_encoded"] = (
        df["race_class"]
        .map(RACE_CLASS_MAP)
        .fillna(0)
        .astype(int)
    )

    df["is_debut_race"] = df["race_class"].isna().astype(int)

    # surface → 数値化
    df["surface_encoded"] = df["surface"].map({
        "芝": 1,
        "ダート": 0,
        "障害": 2
    })

    # race_course → 既存の COURSE_MAP を利用
    df["race_course_encoded"] = (
        df["race_course"]
        .map(COURSE_MAP)
        .fillna(OTHER_COURSE_ID)
        .astype(int)
    )

    # condition → 数値化
    df["condition_encoded"] = df["condition"].map(CONDITION_MAP)

    # ===== ここからフェーズ3用の追加 =====
    df["race_date"] = pd.to_datetime(df["race_date"])

    df["corner_list"] = df["corner"].apply(parse_corners)

    df["first_corner"] = df["corner_list"].apply(
        lambda x: x[0] if len(x) >= 1 else np.nan
    )
    df["last_corner"] = df["corner_list"].apply(
        lambda x: x[-1] if len(x) >= 1 else np.nan
    )

    df["n_corners"] = df["corner_list"].apply(len)

    # corner: "1-3-2-1" のような文字列
    corner_splits = (
        df["corner"]
        .astype(str)
        .str.split("-")
    )

    # 最初と最後のコーナー
    df["first_corner"] = pd.to_numeric(corner_splits.str[0], errors="coerce")
    df["last_corner"]  = pd.to_numeric(corner_splits.str[-1], errors="coerce")

    # 位置変化（負なら押し上げ）
    df["corner_gain"] = df["last_corner"] - df["first_corner"]

    return df

def make_last_race_features(df, n_past=5, race_info=None):
    if len(df) == 0:
        return pd.Series(dtype=float)

    # 直近n走
    df = (
        df.sort_values("race_date", ascending=False)
        .head(n_past)
    )

    # === 実走判定 ===
    is_valid = df["rank"].notna()
    valid_df = df.loc[is_valid]

    n_valid_races = is_valid.sum()
    has_dummy_race = int(n_valid_races < len(df))

    # ===== フェーズ1（既存＋クラス） =====
    rank_last = df["rank"].iloc[0] if len(df) > 0 else np.nan
    past_class_last = df["race_class_encoded"].iloc[0] if len(df) > 0 else np.nan

    features = {
        "n_past_races": len(df),
        "n_valid_races": n_valid_races,
        "has_dummy_race": has_dummy_race,

        "rank_mean": valid_df["rank"].mean(),
        "rank_min": valid_df["rank"].min(),
        "rank_last": rank_last,

        "time_diff_mean": valid_df["time_difference"].mean(),
        "last3f_mean": valid_df["last3f"].mean(),

        "distance_mean": valid_df["distance"].mean(),
        "corner_mean": valid_df["last_corner"].mean(),

        "body_weight_mean": valid_df["body_weight"].mean(),
        "body_weight_diff_mean": valid_df["body_weight_diff"].mean(),
        "jockey_weight_mean": valid_df["jockey_weight"].mean(),

        # クラス系
        "past_class_mean": valid_df["race_class_encoded"].mean(),
        "past_class_max": valid_df["race_class_encoded"].max(),
        "past_class_last": past_class_last,
    }

    # クラス差分（すでに入れているはず）
    if race_info is not None and len(valid_df) > 0:
        cur_class = race_info["race_class_encoded"]
        features.update({
            "class_diff_mean": valid_df["race_class_encoded"].mean() - cur_class,
            "class_diff_last": past_class_last - cur_class,
        })
    else:
        features.update({
            "class_diff_mean": np.nan,
            "class_diff_last": np.nan,
        })

    # ===== フェーズ2（条件適性） =====
    if race_info is not None:
        cur_distance = race_info["distance"]
        cur_surface = race_info["surface_encoded"]
        cur_course  = race_info["race_course_encoded"]
        cur_condition = race_info["condition_encoded"]

        # 距離適性
        same_dist = valid_df["distance"].between(cur_distance - 200, cur_distance + 200)
        dist_diff = valid_df["distance"] - cur_distance

        features.update({
            "past_same_distance_rate": same_dist.mean(),
            "past_distance_diff_mean": dist_diff.mean(),
        })

        # 芝ダ適性
        same_surface = valid_df["surface_encoded"] == cur_surface
        features.update({
            "past_same_surface_rate": same_surface.mean(),
            "past_surface_rank_mean": valid_df.loc[same_surface, "rank"].mean(),
        })

        # コース適性
        same_course = valid_df["race_course_encoded"] == cur_course
        features.update({
            "past_same_course_rate": same_course.mean(),
        })

        # 馬場適性
        same_condition = valid_df["condition_encoded"] == cur_condition
        features.update({
            "past_same_condition_rate": same_condition.mean(),
        })

    # ===== ここからフェーズ3 =====

    # ===== ① 安定性 =====
    features.update({
        # 着順・着差のばらつき
        "rank_std": valid_df["rank"].std(),
        "time_diff_std": valid_df["time_difference"].std(),

        # 安定して上位に来ているか
        "top3_rate": (valid_df["rank"] <= 3).mean(),

        # 大崩れ率（10着以下）
        "bad_finish_rate": (valid_df["rank"] >= 10).mean(),
    })

        # ===== ② 成長トレンド =====
    if len(valid_df) >= 2:
        # 新しい順に 0,1,2,...
        x = np.arange(len(valid_df))

        # 着順トレンド（負なら良化）
        try:
            rank_trend = np.polyfit(x, valid_df["rank"], 1)[0]
        except Exception:
            rank_trend = np.nan

        # 着差トレンド（負なら良化）
        try:
            time_diff_trend = np.polyfit(x, valid_df["time_difference"], 1)[0]
        except Exception:
            time_diff_trend = np.nan

        # 直近と平均との差
        rank_improve = valid_df["rank"].iloc[0] - valid_df["rank"].mean()
        time_diff_improve = (
            valid_df["time_difference"].iloc[0]
            - valid_df["time_difference"].mean()
        )
    else:
        rank_trend = np.nan
        time_diff_trend = np.nan
        rank_improve = np.nan
        time_diff_improve = np.nan

    features.update({
        "rank_trend": rank_trend,
        "time_diff_trend": time_diff_trend,
        "rank_improve_from_mean": rank_improve,
        "time_diff_improve_from_mean": time_diff_improve,
    })

    # ===== ③ 脚質（詳細） =====
    corner = valid_df["last_corner"]
    n_horses = valid_df["num_horses"]

    # 先行判定閾値（上位30%）
    front_threshold = np.ceil(n_horses * 0.3)

    # コーナー順位（平均との差）
    corner_rank_in_race = corner.mean()

    # 前後馬数
    n_horses_ahead = corner - 1
    n_horses_behind = n_horses - corner

    # 先行馬数
    n_front_horses = front_threshold

    # 前にいる馬の密度
    front_density = n_horses_ahead / n_horses

    # 前との差（平均との差）
    front_margin = corner - front_threshold

    features.update({
        # 基本
        "corner_rank_mean": corner_rank_in_race,

        # 前後関係
        "n_horses_ahead_mean": n_horses_ahead.mean(),
        "n_horses_behind_mean": n_horses_behind.mean(),

        # 先行度
        "front_rate": (corner <= front_threshold).mean(),
        "front_density_mean": front_density.mean(),

        # どれくらい前に行く馬か
        "front_margin_mean": front_margin.mean(),
    })

    # ===== 脚質：位置＋動き =====
    fc = valid_df["first_corner"]
    lc = valid_df["last_corner"]
    n_horses = valid_df["num_horses"]

    # 順位変化（＋なら捲り）
    corner_gain = fc - lc

    # 正規化（頭数依存を消す）
    corner_gain_rate = corner_gain / n_horses

    # 明確に動いたか（2頭以上）
    is_mover = (corner_gain >= 2).mean()

    # 序盤位置（前にいた割合）
    early_position_rate = (fc <= n_horses * 0.3).mean()

    features.update({
        # 位置
        "first_corner_mean": fc.mean(),
        "last_corner_mean": lc.mean(),

        # 動き
        "corner_gain_mean": corner_gain.mean(),
        "corner_gain_rate_mean": corner_gain_rate.mean(),

        # タイプ判定
        "is_mover_rate": is_mover,
        "early_position_rate": early_position_rate,
    })

    # ===== 脚質 × 距離短縮／延長 =====
    if race_info is not None:
        # 平均過去距離との差（マイナス＝短縮、プラス＝延長）
        distance_shift = features.get("past_distance_diff_mean")

        early_rate = features.get("early_position_rate")
        corner_gain_mean = features.get("corner_gain_mean")

        features.update({
            # 先行型 × 距離短縮
            "early_x_shortening": early_rate * (-distance_shift),

            # 差し・捲り型 × 距離延長
            # corner_gain_mean は負が良いので符号反転
            "mover_x_extension": (-corner_gain_mean) * distance_shift,
        })
    
    # ===== 脚質 × 頭数 =====
    if race_info is not None:
        n_horses = race_info["number_of_horse"]

        early_rate = features.get("early_position_rate")
        corner_gain_mean = features.get("corner_gain_mean")

        features.update({
            # 先行型 × 少頭数
            "early_x_small_field": early_rate / n_horses,

            # 差し・捲り型 × 多頭数
            "mover_x_large_field": (-corner_gain_mean) * n_horses,
        })

    # ===== 擬似・展開圧 =====
    if race_info is not None:
        n_horses = race_info["number_of_horse"]
        cur_distance = race_info["distance"]

        early_rate = features.get("early_position_rate")

        features.update({
            "pace_pressure_index": early_rate * n_horses / cur_distance,
        })

    # ===== ローテ・間隔系（NaT / 新馬対応）=====
    dates = df["race_date"].dropna()

    if len(dates) >= 2:
        # 直近が index=0 になるように並んでいる前提
        intervals = dates.diff(-1).dt.days.iloc[:-1]

        if len(intervals) > 0:
            features.update({
                # 直近ローテ
                "days_since_last_race": intervals.iloc[0],

                # 平均・最小ローテ
                "interval_mean": intervals.mean(),
                "interval_min": intervals.min(),

                # 詰めローテ率（中2週以内 = 14日）
                "short_interval_rate": (intervals <= 14).mean(),

                # 長期休養明けフラグ（8週以上 = 56日）
                "long_rest_flag": int(intervals.iloc[0] >= 56),
            })
        else:
            features.update({
                "days_since_last_race": np.nan,
                "interval_mean": np.nan,
                "interval_min": np.nan,
                "short_interval_rate": np.nan,
                "long_rest_flag": np.nan,
            })
    else:
        features.update({
            "days_since_last_race": np.nan,
            "interval_mean": np.nan,
            "interval_min": np.nan,
            "short_interval_rate": np.nan,
            "long_rest_flag": np.nan,
        })
    
    # ===== クラス重み付き評価 =====
    if len(valid_df) > 0:
        class_weight = valid_df["race_class_encoded"] / 9.0

        weighted_rank = valid_df["rank"] * (1 / (class_weight + 0.1))
        weighted_time_diff = valid_df["time_difference"] * (1 / (class_weight + 0.1))

        top3 = (valid_df["rank"] <= 3).astype(float)
        weighted_top3 = top3 * class_weight

        features.update({
            "rank_mean_weighted_by_class": weighted_rank.mean(),
            "rank_last_weighted_by_class": weighted_rank.iloc[0],
            "time_diff_mean_weighted_by_class": weighted_time_diff.mean(),
            "top3_rate_weighted_by_class": weighted_top3.mean(),
        })
    else:
        features.update({
            "rank_mean_weighted_by_class": np.nan,
            "rank_last_weighted_by_class": np.nan,
            "time_diff_mean_weighted_by_class": np.nan,
            "top3_rate_weighted_by_class": np.nan,
        })

    return pd.Series(features)

def attach_class_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 現在レースのクラスとの差
    df["class_diff_mean"] = df["race_class_encoded"] - df["past_class_mean"]
    df["class_diff_last"] = df["race_class_encoded"] - df["past_class_last"]

    return df

def build_last_race_features(last_races_df, race_info_df):
    last_races_df = preprocess_last_races(last_races_df)

    race_info = race_info_df.iloc[0]

    features = (
        last_races_df
        .groupby("horse_no", group_keys=False)
        .apply(make_last_race_features, race_info=race_info, include_groups=False)
        .reset_index()
    )

    return features

def attach_race_info(df, race_info_df):
    assert len(race_info_df) == 1

    race_info = race_info_df.iloc[0]
    for col in race_info_df.columns:
        df[col] = race_info[col]

    return df

def make_target(result_df):
    df = result_df.copy()

    # rank を数値に変換（失敗したら NaN）
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # 目的変数
    df["target_top3"] = (df["rank"] <= 3).astype("float")
    df["target_win"] = (df["rank"] == 1).astype("float")

    # 競争除外・中止は NaN にする
    df.loc[df["rank"].isna(), ["target_top3", "target_win"]] = np.nan

    return df[["horse_no", "target_top3", "target_win"]]