---
title: Jmoji Human Evaluation
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Jmoji 人手評価システム

日本語テキスト→絵文字翻訳モデルの人手評価を行うアプリケーション。

## 概要

このアプリケーションでは、日本語テキストに対する絵文字翻訳の品質を評価します。

- **評価対象**: 教師出力（Gold）、モデルA（focal_top50）、モデルB（top50）
- **評価項目**:
  - 意味的一致度（0-4）
  - 自然さ（0-4）
  - 誤解の可能性（Yes/No）
  - モデル比較（A/B/同等）

## 使い方

1. 表示されるテキストと絵文字出力を確認
2. 各出力について評価項目を選択
3. 「次へ」ボタンで次のサンプルへ移動
4. 最後のサンプルで「評価を送信」ボタンをクリック

## 関連リンク

- [Jmojiプロジェクト](https://github.com/AtefAndrus/Jmoji)
- [データセット](https://huggingface.co/datasets/AtefAndrus/jmoji-dataset)
