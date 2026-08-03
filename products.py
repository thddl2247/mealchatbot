import re
from typing import Iterable

import pandas as pd

DISPLAY_COLUMNS = [
    "product_id",
    "상품명",
    "카테고리",
    "세부카테고리",
    "주요성분",
    "가격",
    "용량",
    "판매상태",
    "재고수량",
    "배송유형",
    "배송가능",
    "맵기",
    "평점",
    "리뷰수",
    "추천대상",
    "메뉴태그",
    "상황태그",
    "알레르기정보",
]


def _pattern(values: Iterable[str] | None) -> str:
    return "|".join(re.escape(str(value).strip()) for value in (values or []) if str(value).strip())


def normalize_products(df: pd.DataFrame) -> pd.DataFrame:
    """CSV 열의 숫자·문자 형식을 검색에 적합하게 정리한다."""
    normalized = df.copy()
    required = {
        "product_id", "상품명", "카테고리", "세부카테고리", "주요성분", "가격", "용량",
        "판매상태", "재고수량", "배송유형", "배송가능", "맵기", "평점", "리뷰수",
        "추천대상", "메뉴태그", "상황태그", "알레르기정보",
    }
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"상품 데이터에 필요한 열이 없습니다: {', '.join(missing)}")

    normalized["가격"] = pd.to_numeric(normalized["가격"], errors="coerce").fillna(0).astype(int)
    normalized["재고수량"] = pd.to_numeric(normalized["재고수량"], errors="coerce").fillna(0).astype(int)
    normalized["평점"] = pd.to_numeric(normalized["평점"], errors="coerce").fillna(0.0)
    normalized["리뷰수"] = pd.to_numeric(normalized["리뷰수"], errors="coerce").fillna(0).astype(int)

    text_columns = required - {"가격", "재고수량", "평점", "리뷰수"}
    for column in text_columns:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    return normalized


def available_products(df: pd.DataFrame, include_alcohol: bool = False) -> pd.DataFrame:
    """판매 중·재고 보유·배송 가능 상품만 반환한다."""
    filtered = df[
        (df["판매상태"] == "판매중")
        & (df["재고수량"] > 0)
        & (df["배송가능"] == "Y")
    ].copy()
    if not include_alcohol:
        filtered = filtered[filtered["카테고리"] != "주류"]
    return filtered


def search_products(
    df: pd.DataFrame,
    keywords: list[str] | None = None,
    category: str | None = None,
    max_price: int | None = None,
    exclude_ingredients: list[str] | None = None,
    mild_only: bool = False,
    include_alcohol: bool = False,
    limit: int = 10,
) -> pd.DataFrame:
    """한 재료와 일치하는 실제 판매 가능 상품을 조회한다."""
    filtered = available_products(df, include_alcohol=include_alcohol)

    if category:
        category_pattern = re.escape(category.strip())
        filtered = filtered[
            filtered["카테고리"].str.contains(category_pattern, case=False, na=False)
            | filtered["세부카테고리"].str.contains(category_pattern, case=False, na=False)
        ]

    keyword_pattern = _pattern(keywords)
    if keyword_pattern:
        name_match = filtered["상품명"].str.contains(keyword_pattern, case=False, na=False)
        ingredient_match = filtered["주요성분"].str.contains(keyword_pattern, case=False, na=False)
        subcategory_match = filtered["세부카테고리"].str.contains(keyword_pattern, case=False, na=False)
        category_match = filtered["카테고리"].str.contains(keyword_pattern, case=False, na=False)
        filtered = filtered[name_match | ingredient_match | subcategory_match | category_match].copy()
        filtered["_relevance"] = (
            name_match.loc[filtered.index].astype(int) * 5
            + subcategory_match.loc[filtered.index].astype(int) * 4
            + ingredient_match.loc[filtered.index].astype(int) * 3
            + category_match.loc[filtered.index].astype(int)
        )
    else:
        filtered["_relevance"] = 0

    exclude_pattern = _pattern(exclude_ingredients)
    if exclude_pattern:
        filtered = filtered[
            ~filtered["알레르기정보"].str.contains(exclude_pattern, case=False, na=False)
            & ~filtered["주요성분"].str.contains(exclude_pattern, case=False, na=False)
            & ~filtered["상품명"].str.contains(exclude_pattern, case=False, na=False)
        ]

    if mild_only:
        filtered = filtered[filtered["맵기"].isin(["", "없음", "순함"])]

    if max_price is not None and max_price > 0:
        filtered = filtered[filtered["가격"] <= max_price]

    # 같은 상품군에서는 평점·리뷰가 높고 가격이 낮은 상품을 우선한다.
    return filtered.sort_values(
        ["_relevance", "평점", "리뷰수", "가격"],
        ascending=[False, False, False, True],
    ).drop(columns=["_relevance"], errors="ignore").head(max(1, min(int(limit or 10), 20)))


def search_menu_products(
    df: pd.DataFrame,
    menu_name: str,
    exclude_ingredients: list[str] | None = None,
    mild_only: bool = False,
    include_alcohol: bool = False,
    limit: int = 20,
) -> pd.DataFrame:
    """메뉴태그를 기준으로 특정 메뉴를 구성할 수 있는 상품 존재 여부를 확인한다."""
    filtered = available_products(df, include_alcohol=include_alcohol)
    menu_pattern = re.escape(menu_name.strip())
    filtered = filtered[
        filtered["메뉴태그"].str.contains(menu_pattern, case=False, na=False)
        | filtered["상품명"].str.contains(menu_pattern, case=False, na=False)
    ]

    exclude_pattern = _pattern(exclude_ingredients)
    if exclude_pattern:
        filtered = filtered[
            ~filtered["알레르기정보"].str.contains(exclude_pattern, case=False, na=False)
            & ~filtered["주요성분"].str.contains(exclude_pattern, case=False, na=False)
        ]
    if mild_only:
        filtered = filtered[filtered["맵기"].isin(["", "없음", "순함"])]

    return filtered.sort_values(
        ["평점", "리뷰수", "가격"], ascending=[False, False, True]
    ).head(max(1, min(int(limit or 20), 30)))


def to_records(df: pd.DataFrame) -> list[dict]:
    """도구 응답에 사용할 JSON 직렬화 가능 레코드로 변환한다."""
    return df[DISPLAY_COLUMNS].fillna("").to_dict(orient="records")
