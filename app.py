"""Jmoji 人手評価アプリ.

日本語テキスト→絵文字翻訳モデルの人手評価を行うGradioアプリケーション。

評価項目:
- 意味的一致度（0-4）: テキストの意味を絵文字が表現しているか
- 自然さ（0-4）: SNSで見かけそうか
- 誤解の可能性（Yes/No）: 元の文と逆の印象を与えるか
- モデル比較（A/B/同等）: どちらのモデル出力が良いか
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitScheduler

# ====== 定数 ======
SAMPLES_PATH = Path("data/samples.jsonl")
RESPONSES_DIR = Path("responses")
RESPONSES_DIR.mkdir(exist_ok=True)

# ====== CommitScheduler設定 ======
# HuggingFace Spaceにデプロイ時のみ有効
try:
    scheduler = CommitScheduler(
        repo_id="AtefAndrus/jmoji-human-eval",
        repo_type="space",
        folder_path=RESPONSES_DIR,
        path_in_repo="responses",
        every=5,  # 5分ごとにコミット
    )
    USE_SCHEDULER = True
except Exception:
    # ローカル実行時はスケジューラなし
    scheduler = None
    USE_SCHEDULER = False


# ====== サンプル読み込み ======
def load_samples() -> list[dict]:
    """評価サンプルを読み込む."""
    samples = []
    if SAMPLES_PATH.exists():
        with open(SAMPLES_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples


SAMPLES = load_samples()
TOTAL_SAMPLES = len(SAMPLES) if SAMPLES else 1


# ====== ヘルパー関数 ======
def get_evaluator_id() -> str:
    """評価者IDを生成（セッションごとにユニーク）."""
    return f"anon_{uuid.uuid4().hex[:8]}"


def save_response(evaluator_id: str, responses: dict) -> None:
    """評価結果をファイルに保存."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"responses_{evaluator_id}_{timestamp}.jsonl"
    filepath = RESPONSES_DIR / filename

    if USE_SCHEDULER and scheduler:
        with scheduler.lock:
            with open(filepath, "w", encoding="utf-8") as f:
                for sample_id, response in responses.items():
                    record = {
                        "evaluator_id": evaluator_id,
                        "sample_id": sample_id,
                        **response,
                        "timestamp": timestamp,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            for sample_id, response in responses.items():
                record = {
                    "evaluator_id": evaluator_id,
                    "sample_id": sample_id,
                    **response,
                    "timestamp": timestamp,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ====== UI構築 ======
with gr.Blocks(
    title="Jmoji 人手評価",
    theme=gr.themes.Soft(),
    css="""
    .emoji-display { font-size: 1.5em; }
    .section-header { margin-top: 1em; margin-bottom: 0.5em; }
    """,
) as demo:
    # セッション状態
    state = gr.State(
        {
            "evaluator_id": None,
            "current_idx": 0,
            "responses": {},
        }
    )

    # ヘッダー
    gr.Markdown("# Jmoji 人手評価システム")
    gr.Markdown(
        "日本語テキスト→絵文字翻訳モデルの評価にご協力ください。"
        "各サンプルについて、教師出力（Gold）と2つのモデル出力を評価してください。"
    )

    # 進捗表示
    progress = gr.Markdown(f"**進捗: 1 / {TOTAL_SAMPLES}**")

    # 入力文表示
    gr.Markdown("## 入力文", elem_classes=["section-header"])
    input_text = gr.Textbox(
        label="評価対象のテキスト",
        interactive=False,
        lines=2,
    )

    # 出力表示（3列）
    gr.Markdown("## 出力の比較", elem_classes=["section-header"])
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 教師出力（Gold）")
            gold_output = gr.Textbox(
                label="Gold",
                interactive=False,
                elem_classes=["emoji-display"],
            )
        with gr.Column():
            gr.Markdown("### モデルA（focal_top50）")
            model_a_output = gr.Textbox(
                label="Model A",
                interactive=False,
                elem_classes=["emoji-display"],
            )
        with gr.Column():
            gr.Markdown("### モデルB（top50）")
            model_b_output = gr.Textbox(
                label="Model B",
                interactive=False,
                elem_classes=["emoji-display"],
            )

    # 評価セクション（3列）
    gr.Markdown("## 評価", elem_classes=["section-header"])
    gr.Markdown(
        "各出力について以下を評価してください："
        "\n- **意味的一致度**: 入力文の意味・ニュアンスを絵文字が表現しているか"
        "\n- **自然さ**: 実際のSNSで見かけそうな使い方か"
        "\n- **誤解の可能性**: 元の文の意図と逆の印象を与えそうか"
    )

    with gr.Row():
        # Gold評価
        with gr.Column():
            gr.Markdown("#### Gold評価")
            gold_semantic = gr.Radio(
                choices=[0, 1, 2, 3, 4],
                label="意味的一致度（0:関係なし → 4:非常に妥当）",
                value=None,
            )
            gold_naturalness = gr.Radio(
                choices=[0, 1, 2, 3, 4],
                label="自然さ（0:不自然 → 4:非常に自然）",
                value=None,
            )
            gold_misleading = gr.Radio(
                choices=["No", "Yes"],
                label="誤解を招く可能性",
                value=None,
            )

        # Model A評価
        with gr.Column():
            gr.Markdown("#### モデルA評価")
            model_a_semantic = gr.Radio(
                choices=[0, 1, 2, 3, 4],
                label="意味的一致度（0:関係なし → 4:非常に妥当）",
                value=None,
            )
            model_a_naturalness = gr.Radio(
                choices=[0, 1, 2, 3, 4],
                label="自然さ（0:不自然 → 4:非常に自然）",
                value=None,
            )
            model_a_misleading = gr.Radio(
                choices=["No", "Yes"],
                label="誤解を招く可能性",
                value=None,
            )

        # Model B評価
        with gr.Column():
            gr.Markdown("#### モデルB評価")
            model_b_semantic = gr.Radio(
                choices=[0, 1, 2, 3, 4],
                label="意味的一致度（0:関係なし → 4:非常に妥当）",
                value=None,
            )
            model_b_naturalness = gr.Radio(
                choices=[0, 1, 2, 3, 4],
                label="自然さ（0:不自然 → 4:非常に自然）",
                value=None,
            )
            model_b_misleading = gr.Radio(
                choices=["No", "Yes"],
                label="誤解を招く可能性",
                value=None,
            )

    # モデル比較
    gr.Markdown("## モデル比較", elem_classes=["section-header"])
    preference = gr.Radio(
        choices=["A（focal_top50）", "B（top50）", "同等"],
        label="どちらのモデル出力が良いですか？",
        value=None,
    )
    comment = gr.Textbox(
        label="コメント（任意）",
        placeholder="気づいた点があればご記入ください",
        lines=2,
    )

    # ナビゲーション
    with gr.Row():
        prev_btn = gr.Button("◀ 前へ", variant="secondary")
        next_btn = gr.Button("次へ ▶", variant="primary")

    # 送信ボタン
    submit_btn = gr.Button("評価を送信", variant="primary", visible=False)

    # ステータス表示
    status_msg = gr.Markdown("")

    # ====== イベントハンドラ ======
    def init_session(state_dict: dict) -> tuple:
        """セッション初期化."""
        if state_dict["evaluator_id"] is None:
            state_dict["evaluator_id"] = get_evaluator_id()

        if not SAMPLES:
            return (
                state_dict,
                "**エラー: サンプルが見つかりません**",
                "サンプルがありません",
                "-",
                "-",
                "-",
                gr.update(visible=False),
                "",
            )

        sample = SAMPLES[0]
        return (
            state_dict,
            f"**進捗: 1 / {TOTAL_SAMPLES}**",
            sample["text"],
            sample["gold"],
            sample.get("pred_focal_top50", "-"),
            sample.get("pred_top50", "-"),
            gr.update(visible=(TOTAL_SAMPLES == 1)),
            "",
        )

    def collect_current_response(
        gold_sem,
        gold_nat,
        gold_mis,
        model_a_sem,
        model_a_nat,
        model_a_mis,
        model_b_sem,
        model_b_nat,
        model_b_mis,
        pref,
        cmt,
    ) -> dict:
        """現在の評価を辞書にまとめる."""
        return {
            "gold": {
                "semantic": gold_sem,
                "naturalness": gold_nat,
                "misleading": gold_mis == "Yes" if gold_mis else None,
            },
            "model_a": {
                "semantic": model_a_sem,
                "naturalness": model_a_nat,
                "misleading": model_a_mis == "Yes" if model_a_mis else None,
            },
            "model_b": {
                "semantic": model_b_sem,
                "naturalness": model_b_nat,
                "misleading": model_b_mis == "Yes" if model_b_mis else None,
            },
            "preference": pref,
            "comment": cmt,
        }

    def restore_response(response: dict) -> tuple:
        """保存済みの評価を復元."""
        gold = response.get("gold", {})
        model_a = response.get("model_a", {})
        model_b = response.get("model_b", {})

        def to_misleading_str(val):
            if val is True:
                return "Yes"
            elif val is False:
                return "No"
            return None

        return (
            gold.get("semantic"),
            gold.get("naturalness"),
            to_misleading_str(gold.get("misleading")),
            model_a.get("semantic"),
            model_a.get("naturalness"),
            to_misleading_str(model_a.get("misleading")),
            model_b.get("semantic"),
            model_b.get("naturalness"),
            to_misleading_str(model_b.get("misleading")),
            response.get("preference"),
            response.get("comment", ""),
        )

    def navigate(
        direction: int,
        state_dict: dict,
        gold_sem,
        gold_nat,
        gold_mis,
        model_a_sem,
        model_a_nat,
        model_a_mis,
        model_b_sem,
        model_b_nat,
        model_b_mis,
        pref,
        cmt,
    ) -> tuple:
        """ページ移動（前へ/次へ）."""
        current_idx = state_dict["current_idx"]

        # 現在の評価を保存
        sample_id = SAMPLES[current_idx]["id"]
        state_dict["responses"][sample_id] = collect_current_response(
            gold_sem,
            gold_nat,
            gold_mis,
            model_a_sem,
            model_a_nat,
            model_a_mis,
            model_b_sem,
            model_b_nat,
            model_b_mis,
            pref,
            cmt,
        )

        # インデックス更新
        new_idx = max(0, min(current_idx + direction, TOTAL_SAMPLES - 1))
        state_dict["current_idx"] = new_idx

        # 新しいサンプル
        sample = SAMPLES[new_idx]
        new_sample_id = sample["id"]

        # 既存の評価を復元
        existing = state_dict["responses"].get(new_sample_id, {})
        restored = restore_response(existing)

        is_last = new_idx == TOTAL_SAMPLES - 1

        return (
            state_dict,
            f"**進捗: {new_idx + 1} / {TOTAL_SAMPLES}**",
            sample["text"],
            sample["gold"],
            sample.get("pred_focal_top50", "-"),
            sample.get("pred_top50", "-"),
            *restored,
            gr.update(visible=is_last),
            "",
        )

    def submit_all(
        state_dict: dict,
        gold_sem,
        gold_nat,
        gold_mis,
        model_a_sem,
        model_a_nat,
        model_a_mis,
        model_b_sem,
        model_b_nat,
        model_b_mis,
        pref,
        cmt,
    ) -> str:
        """全評価を送信."""
        current_idx = state_dict["current_idx"]

        # 最後のサンプルを保存
        sample_id = SAMPLES[current_idx]["id"]
        state_dict["responses"][sample_id] = collect_current_response(
            gold_sem,
            gold_nat,
            gold_mis,
            model_a_sem,
            model_a_nat,
            model_a_mis,
            model_b_sem,
            model_b_nat,
            model_b_mis,
            pref,
            cmt,
        )

        # ファイルに保存
        evaluator_id = state_dict["evaluator_id"]
        save_response(evaluator_id, state_dict["responses"])

        return (
            f"## 評価が完了しました\n\n"
            f"ご協力ありがとうございました。\n\n"
            f"- 評価者ID: `{evaluator_id}`\n"
            f"- 評価サンプル数: {len(state_dict['responses'])}件"
        )

    # イベント接続
    demo.load(
        init_session,
        inputs=[state],
        outputs=[
            state,
            progress,
            input_text,
            gold_output,
            model_a_output,
            model_b_output,
            submit_btn,
            status_msg,
        ],
    )

    eval_inputs = [
        gold_semantic,
        gold_naturalness,
        gold_misleading,
        model_a_semantic,
        model_a_naturalness,
        model_a_misleading,
        model_b_semantic,
        model_b_naturalness,
        model_b_misleading,
        preference,
        comment,
    ]

    all_outputs = [
        state,
        progress,
        input_text,
        gold_output,
        model_a_output,
        model_b_output,
        *eval_inputs,
        submit_btn,
        status_msg,
    ]

    prev_btn.click(
        lambda *args: navigate(-1, *args),
        inputs=[state, *eval_inputs],
        outputs=all_outputs,
    )

    next_btn.click(
        lambda *args: navigate(1, *args),
        inputs=[state, *eval_inputs],
        outputs=all_outputs,
    )

    submit_btn.click(
        submit_all,
        inputs=[state, *eval_inputs],
        outputs=[status_msg],
    )

if __name__ == "__main__":
    demo.launch()
