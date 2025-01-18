import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# 履歴にテキストを追加
def add_to_history(input_text):
    if "text_history" not in st.session_state:
        st.session_state["text_history"] = []
    st.session_state["text_history"].append(input_text)
    st.success(f"「{input_text}」を履歴に追加しました。")
    st.experimental_rerun()


# 履歴を表示
def display_history():
    st.write("### 現在のテキスト履歴:")
    if st.session_state.get("text_history"):
        for i, text in enumerate(st.session_state["text_history"], 1):
            st.write(f"{i}. {text}")
    else:
        st.write("履歴はまだありません。")


# ベクトル化と3次元プロット
def vectorize_and_plot():
    if st.session_state.get("text_history"):
        if len(st.session_state["text_history"]) < 3:
            st.warning("履歴が少なくとも3つ以上必要です。テキストを追加してください。")
            return

        # ベクトル化
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(st.session_state["text_history"])
        feature_names = vectorizer.get_feature_names_out()
        vector_df = pd.DataFrame(vectors.toarray(), columns=feature_names)

        # ベクトルの表示
        st.write("### TF-IDFベクトル:")
        st.dataframe(vector_df)

        # 次元削減 (PCA)
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(vectors.toarray())

        # データフレーム化
        df = pd.DataFrame(reduced_vectors, columns=["x", "y", "z"])
        df["label"] = range(1, len(st.session_state["text_history"]) + 1)

        # 3Dプロット
        st.write("### 3次元空間にマッピングされた結果:")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df["x"], df["y"], df["z"], alpha=0.7)

        # 各点に番号ラベルを追加
        for i, row in df.iterrows():
            ax.text(row["x"], row["y"], row["z"], str(row["label"]), fontsize=8)

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        st.pyplot(fig)
    else:
        st.warning("履歴が空です。テキストを入力してください。")


# メイン処理
def main():
    st.title("テキストデータの3次元ベクトル化アプリ")
    st.write(
        "デフォルトで10個の英文を表示し、それに追加でテキストを入力して3次元空間にプロットします。"
    )

    # 初期化（デフォルトの10個の英文）
    if "text_history" not in st.session_state:
        st.session_state["text_history"] = [
            "I love programming.",
            "Streamlit makes data visualization simple.",
            "Machine learning is fascinating.",
            "Data science involves statistics and programming.",
            "Natural language processing is a part of AI.",
            "I enjoy learning new technologies.",
            "Python is a versatile programming language.",
            "Visualization helps to understand data.",
            "AI is transforming the world.",
            "Collaboration makes projects better.",
        ]

    # テキスト入力
    input_text = st.text_input("テキストを入力してください:")
    if input_text and st.button("履歴に追加"):
        add_to_history(input_text)

    # 履歴表示
    display_history()

    # ベクトル化して3次元プロット
    if st.button("ベクトル化して3次元プロット"):
        vectorize_and_plot()

    # 履歴をクリア
    if st.button("履歴をクリア"):
        st.session_state["text_history"] = []
        st.success("履歴をクリアしました。")


if __name__ == "__main__":
    main()
