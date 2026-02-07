"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import constants as ct
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document


try:
    import pydantic
    if not hasattr(pydantic, "BaseSettings"):
        from pydantic_settings import BaseSettings

        pydantic.BaseSettings = BaseSettings
except Exception:
    pass

from langchain_community.vectorstores import Chroma


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

# Streamlit CloudのSecretsから環境変数を補完
try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)

    # Streamlit Cloudなどでログが見えるように標準出力にも出す
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    # ==========================================
    # ★社員名簿CSVだけ「1ドキュメント化」して、splitしない
    # ==========================================
    roster_docs = []
    other_docs = []

    for d in docs_all:
        # metadata["source"] は loader により無い場合もあるので安全に
        src = (d.metadata.get("source") or "").replace("\\", "/")
        name = Path(src).name

        # ファイル名が違う可能性があるので「社員名簿」が入ってたら対象にする
        if name.endswith(".csv") and "社員名簿" in name:
            roster_docs.append(d)
        else:
            other_docs.append(d)

    # 社員名簿が見つかったら、いったん docs_all から除外して「1つのdoc」に作り直す
    if roster_docs:
        # どのcsvか1つに決める（通常は1ファイルのはず）
        roster_src = (roster_docs[0].metadata.get("source") or "").replace("\\", "/")
        merged_roster = load_employee_roster_as_one_doc(roster_src)
    else:
        merged_roster = []

    # 通常ドキュメントだけチャンク分割
    splitted_other = text_splitter.split_documents(other_docs)

    # ★社員名簿は分割せず混ぜる
    splitted_docs = splitted_other + merged_roster

    # ベクターストアの作成
    PERSIST_DIR = "./.chroma"

    # ベクターストアの作成（永続化）
    db = Chroma.from_documents(
        splitted_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    db.persist()

    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_TOP_K})

    # 互換性のため従来のキーも保持
    st.session_state.retriever = retriever

    # モード別で参照するRetriever（utils.pyと整合）
    st.session_state.retriever_internal = retriever
    st.session_state.retriever_employees = retriever


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        try:
            # 指定のWebページを読み込み
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            # for文の外のリストに読み込んだデータソースを追加
            web_docs_all.extend(web_docs)
        except ModuleNotFoundError as e:
            logger.warning(f"Webページ読み込みをスキップしました（{web_url}）\n{e}")
            continue
        except Exception as e:
            logger.warning(f"Webページ読み込みに失敗しました（{web_url}）\n{e}")
            continue
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s


def load_employee_roster_as_one_doc(csv_path: str):
    """
    社員名簿CSVを1つのDocumentに変換
    """
    df = pd.read_csv(csv_path, encoding="utf-8").fillna("")

    # 部署っぽい列を探す（教材と列名が違っても耐える）
    dept_col = None
    for c in df.columns:
        if "部署" in c or "部門" in c:
            dept_col = c
            break

    index_lines = ["【データ】社員名簿（CSV）"]
    if dept_col:
        depts = sorted(set([str(d) for d in df[dept_col].tolist() if str(d)]))
        index_lines.append("【部署一覧】" + " / ".join(depts))

    # そのまま表にしてLLMが一覧化しやすい形にする
    table_md = df.to_markdown(index=False)

    text = "\n".join(index_lines) + "\n\n【社員一覧】\n" + table_md

    return [Document(page_content=text, metadata={"source": csv_path, "type": "employee_roster"})]


