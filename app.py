import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import APIError, OpenAI

from products import normalize_products, search_menu_products, search_products, to_records
from system_prompt import SYSTEM_PROMPT

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_CSV = "shopping_products_test_700_home_meals.csv"
PRODUCTS_CSV_PATH = os.getenv("PRODUCTS_CSV_PATH", DEFAULT_CSV)
MAX_TOOL_ROUNDS = 12

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "product_id": {"type": "string", "description": "실제 조회된 상품 ID"},
        "ingredient": {"type": "string", "description": "레시피 재료명"},
        "product_name": {"type": "string", "description": "실제 조회된 상품명"},
        "price": {"type": "integer", "description": "상품 1개 가격(원)"},
        "unit": {"type": "string", "description": "용량/단위"},
        "quantity": {"type": "integer", "minimum": 1, "description": "추천 수량"},
        "delivery_type": {"type": "string", "description": "배송유형"},
        "sale_status": {"type": "string", "description": "판매상태"},
    },
    "required": ["product_id", "ingredient", "product_name", "price", "quantity"],
}

MENU_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "구체적인 요리명"},
        "description": {"type": "string", "description": "메뉴 특징 한 줄"},
        "reason": {"type": "string", "description": "사용자 조건에 맞는 이유"},
        "cook_time_minutes": {"type": "integer", "description": "예상 조리시간"},
        "difficulty": {"type": "string", "description": "쉬움/보통/어려움"},
        "estimated_budget": {"type": "integer", "description": "대략적인 예상 재료비"},
    },
    "required": ["name", "reason"],
}

RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "실제 재료명(한글)"},
                    "essential": {"type": "boolean", "description": "이 요리에 꼭 필요한 재료인지 여부"},
                    "note": {"type": "string", "description": "대략적인 분량이나 비고"},
                },
                "required": ["name", "essential"],
            },
        },
    },
    "required": ["ingredients"],
}


def fetch_recipe_ingredients(menu_name: str, servings: int = 2) -> list[dict[str, Any]]:
    """실제 요리 레시피 기준으로 필요한 재료 목록을 OpenAI에 별도로 질의해 가져온다."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 정확한 요리 레시피 전문가입니다. 실제로 그 요리를 만들 때 "
                        "필요한 재료만 빠짐없이 정확한 재료명으로 나열합니다. "
                        "존재하지 않는 재료를 지어내지 않고, 실제 조리 과정에 쓰이는 "
                        "조미료·향신료·부재료까지 빠짐없이 포함합니다."
                    ),
                },
                {
                    "role": "user",
                    "content": f"'{menu_name}' {servings}인분 기준 실제 레시피에 필요한 재료를 모두 알려줘.",
                },
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "return_recipe_ingredients",
                    "description": "레시피 재료 목록을 구조화된 형식으로 반환합니다.",
                    "parameters": RECIPE_SCHEMA,
                },
            }],
            tool_choice={"type": "function", "function": {"name": "return_recipe_ingredients"}},
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        args = json.loads(message.tool_calls[0].function.arguments or "{}")
        return args.get("ingredients", [])
    except (APIError, json.JSONDecodeError):
        return []


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_menu_products",
            "description": "메뉴 후보를 제시하기 전에 메뉴태그 기준으로 실제 판매 가능 상품이 존재하는지 확인합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_name": {"type": "string", "description": "확인할 구체적인 메뉴명 하나"},
                    "exclude_ingredients": {"type": "array", "items": {"type": "string"}},
                    "mild_only": {"type": "boolean"},
                    "include_alcohol": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["menu_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe_ingredients",
            "description": (
                "메뉴가 확정되면 재료를 추측하지 말고 이 도구를 먼저 호출해 실제 요리 레시피 기준 "
                "재료 목록을 가져옵니다. 이 결과에 있는 재료명만 search_products 조회에 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_name": {"type": "string", "description": "확정된 구체적인 메뉴명"},
                    "servings": {"type": "integer", "description": "인원 수(기본 2)"},
                },
                "required": ["menu_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "확정된 메뉴의 재료 하나에 맞는 실제 판매 가능 상품을 조회합니다. "
                "서로 다른 재료를 한 호출에 섞지 말고, 조회 결과 안에서만 최종 상품을 선택하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string"},
                    "max_price": {"type": "integer"},
                    "exclude_ingredients": {"type": "array", "items": {"type": "string"}},
                    "mild_only": {"type": "boolean"},
                    "include_alcohol": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "present_menu_options",
            "description": "메뉴가 미정인 사용자에게 선택 가능한 메뉴 후보를 동일한 카드 UI로 보여줍니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "condition_chips": {"type": "array", "items": {"type": "string"}},
                    "menus": {"type": "array", "minItems": 2, "maxItems": 3, "items": MENU_SCHEMA},
                },
                "required": ["summary", "condition_chips", "menus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "present_recommendation",
            "description": "확정된 메뉴의 실제 상품 장바구니 구성을 카드 UI로 보여줍니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "condition_chips": {"type": "array", "items": {"type": "string"}},
                    "menu": MENU_SCHEMA,
                    "required_items": {"type": "array", "items": ITEM_SCHEMA},
                    "pantry_check_items": {"type": "array", "items": {"type": "string"}},
                    "optional_items": {"type": "array", "items": ITEM_SCHEMA},
                    "missing_ingredients": {"type": "array", "items": {"type": "string"}},
                    "total_price": {"type": "integer"},
                    "budget_remaining": {"type": "integer"},
                },
                "required": [
                    "summary", "condition_chips", "menu", "required_items",
                    "total_price", "budget_remaining",
                ],
            },
        },
    },
]

CARD_CSS = """
<style>
:root { --green:#276554; --dark:#173f35; --light:#eef8f3; --line:#e3eae7; }
.block-container { padding-top: 1.6rem; max-width: 900px; }
[data-testid="stChatMessage"] { border: 0; padding: .35rem 0; }
.recommend-shell { background:#fff; border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 12px 32px rgba(23,63,53,.08); }
.chat-summary { font-size:1rem; font-weight:750; color:var(--dark); line-height:1.55; margin-bottom:13px; }
.chip-row { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 15px; }
.chip { background:#e9f5f0; color:var(--green); border-radius:999px; padding:5px 11px; font-size:.75rem; font-weight:700; }
.menu-option, .menu-card { background:linear-gradient(135deg,#f7fbf9,#edf7f2); border:1px solid #d8e9e1; border-radius:17px; padding:15px 17px; margin:10px 0; }
.menu-title { font-size:1.12rem; font-weight:800; color:var(--dark); }
.menu-meta { color:#6b7772; font-size:.78rem; margin:5px 0 8px; }
.menu-reason { color:#315f51; font-size:.86rem; line-height:1.45; }
.section-title { font-size:.9rem; font-weight:800; color:var(--dark); margin:17px 0 8px; }
.item-row { border:1px solid var(--line); border-radius:14px; padding:11px 12px; margin-bottom:8px; background:#fff; }
.item-name { font-weight:800; color:var(--dark); font-size:.86rem; }
.item-product { color:#3f4b47; font-size:.79rem; margin-top:2px; }
.item-meta { color:#77827e; font-size:.72rem; margin-top:3px; }
.total-box { background:#f6f8f7; border-radius:14px; padding:12px 14px; margin-top:14px; }
.total-line { display:flex; justify-content:space-between; padding:3px 0; font-size:.88rem; }
.total-line strong { color:var(--dark); }
.over { color:#c0392b !important; }
.help-note { background:#fff8e8; color:#755e24; border-radius:12px; padding:10px 12px; font-size:.78rem; line-height:1.45; }
div[data-testid="stButton"] button { width:100%; border-radius:12px; font-weight:750; min-height:42px; }
.small-muted { color:#71807a; font-size:.76rem; }
</style>
"""

st.set_page_config(page_title="담이 AI 장보기", page_icon="🛒", layout="centered")
st.markdown(CARD_CSS, unsafe_allow_html=True)
st.title("🛒 담이 AI 장보기 도우미")
st.caption("메뉴 추천부터 실제 테스트 상품 장바구니 구성까지 연결합니다.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()


@st.cache_data
def load_products(path: str) -> pd.DataFrame:
    return normalize_products(pd.read_csv(path, encoding="utf-8-sig"))


csv_path = Path(PRODUCTS_CSV_PATH)
if not csv_path.exists():
    fallback = Path(__file__).resolve().parent / PRODUCTS_CSV_PATH
    csv_path = fallback

try:
    products_df = load_products(str(csv_path))
except (FileNotFoundError, ValueError) as error:
    st.error(f"상품 데이터 로드 실패: {error}")
    st.stop()

client = OpenAI(api_key=api_key)

DEFAULT_STATE = {
    "messages": [],
    "cart": {},
    "active_panel": None,
    "active_panel_anchor": None,
    "owned_product_ids": set(),
    "owned_pantry_items": set(),
    "recommended_menu_history": [],
    "is_processing": False,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default.copy() if isinstance(default, (dict, list, set)) else default


def api_history() -> list[dict[str, str]]:
    """OpenAI에 시스템 프롬프트와 전체 대화 내역을 안전한 형식으로 보낸다."""
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": str(content)})
    if st.session_state.recommended_menu_history:
        history.append({
            "role": "system",
            "content": "이 대화에서 이미 추천한 메뉴: " + ", ".join(st.session_state.recommended_menu_history),
        })
    return history


def clean_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": str(item.get("product_id", "")),
        "ingredient": str(item.get("ingredient", "")),
        "product_name": str(item.get("product_name", "")),
        "price": int(item.get("price") or 0),
        "unit": str(item.get("unit", "")),
        "quantity": max(1, int(item.get("quantity") or 1)),
        "delivery_type": str(item.get("delivery_type", "")),
        "sale_status": str(item.get("sale_status", "")),
    }


def normalize_panel(panel: dict[str, Any]) -> dict[str, Any]:
    if panel.get("type") == "recommendation":
        data = panel.get("data", {})
        data["required_items"] = [clean_item(item) for item in data.get("required_items", [])]
        data["optional_items"] = [clean_item(item) for item in data.get("optional_items", [])]
        panel["data"] = data
    return panel


def process_user_turn(user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    messages = api_history()
    st.session_state.is_processing = True

    try:
        final_panel = None
        assistant_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                assistant_text = message.content or ""
                break

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tool_call.model_dump() for tool_call in message.tool_calls],
            })

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "search_menu_products":
                    result = search_menu_products(products_df, **args)
                    tool_content = json.dumps(to_records(result), ensure_ascii=False)
                elif name == "get_recipe_ingredients":
                    ingredients = fetch_recipe_ingredients(
                        args.get("menu_name", ""), int(args.get("servings") or 2)
                    )
                    tool_content = json.dumps({"ingredients": ingredients}, ensure_ascii=False)
                elif name == "search_products":
                    result = search_products(products_df, **args)
                    tool_content = json.dumps(to_records(result), ensure_ascii=False)
                elif name == "present_menu_options":
                    final_panel = {"type": "menu_options", "data": args}
                    tool_content = json.dumps({"status": "rendered"}, ensure_ascii=False)
                elif name == "present_recommendation":
                    final_panel = normalize_panel({"type": "recommendation", "data": args})
                    tool_content = json.dumps({"status": "rendered"}, ensure_ascii=False)
                else:
                    tool_content = json.dumps({"error": "unknown tool"}, ensure_ascii=False)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                })

            if final_panel is not None:
                break

        if final_panel:
            st.session_state.active_panel = final_panel
            data = final_panel.get("data", {})
            summary = data.get("summary", "추천 결과를 준비했어요.")
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.active_panel_anchor = len(st.session_state.messages) - 1
            if final_panel["type"] == "menu_options":
                for menu in data.get("menus", []):
                    name = menu.get("name")
                    if name and name not in st.session_state.recommended_menu_history:
                        st.session_state.recommended_menu_history.append(name)
            st.session_state.owned_product_ids = set()
            st.session_state.owned_pantry_items = set()
        elif assistant_text:
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    except APIError as error:
        st.session_state.messages.append({"role": "assistant", "content": f"API 호출 오류: {error}"})
    except Exception as error:
        st.session_state.messages.append({"role": "assistant", "content": f"처리 중 오류가 발생했습니다: {error}"})
    finally:
        st.session_state.is_processing = False


def add_items_to_cart(items: list[dict[str, Any]]) -> None:
    """같은 상품은 새 행을 만들지 않고 수량을 합산한다."""
    for raw in items:
        item = clean_item(raw)
        product_id = item["product_id"] or item["product_name"]
        if not product_id or product_id in st.session_state.owned_product_ids:
            continue
        if product_id in st.session_state.cart:
            st.session_state.cart[product_id]["quantity"] += item["quantity"]
        else:
            st.session_state.cart[product_id] = item


def selected_recommendation_items(data: dict[str, Any], include_optional: bool) -> list[dict[str, Any]]:
    items = list(data.get("required_items", []))
    if include_optional:
        items += list(data.get("optional_items", []))
    return [item for item in items if item.get("product_id") not in st.session_state.owned_product_ids]


def render_chips(chips: list[str]) -> None:
    if not chips:
        return
    chip_html = "".join(f'<span class="chip">{html.escape(str(chip))}</span>' for chip in chips)
    st.markdown(f'<div class="chip-row">{chip_html}</div>', unsafe_allow_html=True)


def render_menu_options(data: dict[str, Any]) -> None:
    st.markdown('<div class="recommend-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-summary">{html.escape(data.get("summary", ""))}</div>', unsafe_allow_html=True)
    render_chips(data.get("condition_chips", []))

    menus = data.get("menus", [])
    for index, menu in enumerate(menus):
        meta = []
        if menu.get("cook_time_minutes"):
            meta.append(f'{menu["cook_time_minutes"]}분')
        if menu.get("difficulty"):
            meta.append(str(menu["difficulty"]))
        if menu.get("estimated_budget"):
            meta.append(f'약 {int(menu["estimated_budget"]):,}원')
        st.markdown(
            '<div class="menu-option">'
            f'<div class="menu-title">{html.escape(menu.get("name", ""))}</div>'
            f'<div class="menu-meta">{" · ".join(meta)}</div>'
            f'<div class="menu-reason">{html.escape(menu.get("reason", ""))}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button(f'{menu.get("name", "메뉴")} 선택', key=f'menu_select_{index}', use_container_width=True):
            process_user_turn(f'메뉴로 {menu.get("name", "")}을 선택할게. 이 메뉴의 실제 레시피 재료를 상품 데이터에서 찾아 장바구니 구성을 보여줘.')
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_item_selector(items: list[dict[str, Any]], title: str, prefix: str) -> None:
    if not items:
        return
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    for index, item in enumerate(items):
        product_id = item.get("product_id") or f'{prefix}_{index}'
        checked = st.checkbox(
            f'집에 있음 — {item.get("ingredient", "재료")}',
            value=product_id in st.session_state.owned_product_ids,
            key=f'owned_{prefix}_{product_id}_{index}',
            help="체크하면 장바구니 담기 대상에서 제외됩니다.",
        )
        if checked:
            st.session_state.owned_product_ids.add(product_id)
        else:
            st.session_state.owned_product_ids.discard(product_id)

        quantity = int(item.get("quantity") or 1)
        line_total = int(item.get("price") or 0) * quantity
        meta = [f'{line_total:,}원', str(item.get("unit", "")), f'{quantity}개']
        if item.get("delivery_type"):
            meta.append(str(item["delivery_type"]))
        st.markdown(
            '<div class="item-row">'
            f'<div class="item-name">{html.escape(item.get("ingredient", ""))}</div>'
            f'<div class="item-product">{html.escape(item.get("product_name", ""))}</div>'
            f'<div class="item-meta">{" · ".join(html.escape(v) for v in meta if v)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def current_total(data: dict[str, Any], include_optional: bool = True) -> int:
    return sum(
        int(item.get("price") or 0) * int(item.get("quantity") or 1)
        for item in selected_recommendation_items(data, include_optional)
    )


def render_recommendation(data: dict[str, Any]) -> None:
    st.markdown('<div class="recommend-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-summary">{html.escape(data.get("summary", ""))}</div>', unsafe_allow_html=True)
    render_chips(data.get("condition_chips", []))

    menu = data.get("menu", {})
    meta = []
    if menu.get("cook_time_minutes"):
        meta.append(f'{menu["cook_time_minutes"]}분')
    if menu.get("difficulty"):
        meta.append(str(menu["difficulty"]))
    st.markdown(
        '<div class="menu-card">'
        f'<div class="menu-title">🍽 {html.escape(menu.get("name", ""))}</div>'
        f'<div class="menu-meta">{" · ".join(meta)}</div>'
        f'<div class="menu-reason">{html.escape(menu.get("reason", ""))}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="help-note">이미 가지고 있는 재료는 “집에 있음”을 체크하세요. 체크한 상품은 장바구니에서 자동 제외됩니다.</div>', unsafe_allow_html=True)
    render_item_selector(data.get("required_items", []), "필수 재료", "required")

    pantry = data.get("pantry_check_items", [])
    if pantry:
        st.markdown('<div class="section-title">집에 있는지 확인할 기본 재료</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for index, pantry_item in enumerate(pantry):
            checked = cols[index % 2].checkbox(
                str(pantry_item),
                value=str(pantry_item) in st.session_state.owned_pantry_items,
                key=f'pantry_{index}_{pantry_item}',
            )
            if checked:
                st.session_state.owned_pantry_items.add(str(pantry_item))
            else:
                st.session_state.owned_pantry_items.discard(str(pantry_item))

    render_item_selector(data.get("optional_items", []), "선택 재료", "optional")

    missing = data.get("missing_ingredients", [])
    if missing:
        st.warning("판매 가능한 상품을 찾지 못한 재료: " + ", ".join(map(str, missing)))

    budget_remaining = int(data.get("budget_remaining") or 0)
    adjusted_total = current_total(data, include_optional=True)
    st.markdown(
        '<div class="total-box">'
        f'<div class="total-line"><span>현재 선택 상품 합계</span><strong>{adjusted_total:,}원</strong></div>'
        f'<div class="total-line"><span>초기 예산 잔액</span><strong class="{"over" if budget_remaining < 0 else ""}">{budget_remaining:,}원</strong></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    if col1.button("필수 상품 담기", use_container_width=True):
        add_items_to_cart(selected_recommendation_items(data, include_optional=False))
        st.toast("보유 재료를 제외한 필수 상품을 담았습니다.")
        st.rerun()
    if col2.button("선택 상품까지 전체 담기", use_container_width=True):
        add_items_to_cart(selected_recommendation_items(data, include_optional=True))
        st.toast("보유 재료를 제외한 상품을 모두 담았습니다.")
        st.rerun()

    if st.button("다른 메뉴 추천받기", use_container_width=True):
        # active_panel을 지우지 않으므로 API 처리 중에도 동일한 UI 위치가 유지된다.
        process_user_turn("지금까지의 인원, 예산, 상황, 알레르기, 제외 재료 조건은 그대로 유지하고 이전에 추천한 메뉴를 제외한 다른 메뉴 3개를 추천해줘.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("🛒 장바구니")
    cart = st.session_state.cart
    if not cart:
        st.caption("담긴 상품이 없습니다.")
    else:
        cart_total = 0
        for product_id, item in list(cart.items()):
            st.markdown(f'**{item.get("product_name", "")}**')
            col_minus, col_qty, col_plus, col_remove = st.columns([1, 1, 1, 1.3])
            if col_minus.button("−", key=f'minus_{product_id}'):
                item["quantity"] -= 1
                if item["quantity"] <= 0:
                    del cart[product_id]
                st.rerun()
            col_qty.write(str(item.get("quantity", 1)))
            if col_plus.button("+", key=f'plus_{product_id}'):
                item["quantity"] += 1
                st.rerun()
            if col_remove.button("삭제", key=f'remove_{product_id}'):
                del cart[product_id]
                st.rerun()
            line_total = int(item.get("price") or 0) * int(item.get("quantity") or 1)
            cart_total += line_total
            st.caption(f'{item.get("unit", "")} · {line_total:,}원')
            st.divider()
        st.markdown(f"### 합계 {cart_total:,}원")
        if st.button("장바구니 비우기", use_container_width=True):
            st.session_state.cart = {}
            st.rerun()

    st.caption("테스트 상품 데이터를 사용하는 포트폴리오용 프로토타입입니다.")

# 카드 UI는 이를 만든 요약 메시지와 같은 채팅 말풍선 안에 렌더링해 대화 흐름을 유지한다.
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if st.session_state.active_panel and index == st.session_state.active_panel_anchor:
            panel = st.session_state.active_panel
            if panel.get("type") == "menu_options":
                render_menu_options(panel.get("data", {}))
            elif panel.get("type") == "recommendation":
                render_recommendation(panel.get("data", {}))

user_input = st.chat_input("식사 상황이나 정해진 메뉴를 입력하세요.", disabled=st.session_state.is_processing)
if user_input:
    process_user_turn(user_input)
    st.rerun()
