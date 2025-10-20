# build_embedding.py
import pandas as pd 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ----------------- KOD BAŞLANGICI -----------------

# 1. Veri Okuma
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("HATA: 'dataset.csv' dosyası bulunamadı. Lütfen dosya adını ve yolunu kontrol edin.")
    exit()

# 2. Embedding Modelini Yükleme
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Metinleri Birleştirme
texts = (
    df["ekstrat"].astype(str)
    + "|Hücre hattı:" + df["hücre_hattı"].astype(str)
    + "|Konsantrasyon(uM):" + df["konsantrasyon_uM"].astype(str)
    + "|Tahlil:" + df["tahlil"].astype(str)
    + "|Hücre canlılığı (%):" + df["hücre_canlılığı_yüzdesi"].astype(str)
    + "|Not:" + df["notlar"].astype(str)
).tolist()

# 4. Embedding Oluşturma
print("Embedding oluşturuluyor, lütfen bekleyiniz.")
embeddings = model.encode(texts, show_progress_bar=True) 
print("Embedding oluşturma tamamlandı.")

# 5. ChromaDB İstemcisini ve Koleksiyonu Oluşturma
CHROMA_PATH = "./chroma_db"
# PersistentClient ile kalıcı veritabanı oluşturma
client = chromadb.PersistentClient(path=CHROMA_PATH) 

COLLECTION_NAME = "ekstrat"
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Mevcut '{COLLECTION_NAME}' koleksiyonu silindi.")
except Exception:
    pass

collection = client.create_collection(COLLECTION_NAME)
print(f"'{COLLECTION_NAME}' adında yeni bir ChromaDB koleksiyonu oluşturuldu.")

# 6. Verileri Koleksiyona Ekleme
try:
    collection.add(
        ids=[str(i) for i in df["id"]],
        documents=texts,
        # numpy array'i listeye çevirme
        embeddings=embeddings.tolist(),
        metadatas=df.to_dict(orient="records")
    )
    print("\n✅ Embedding'ler başarıyla oluşturuldu ve 'chroma_db' klasörüne kaydedildi.")
    print(f"Toplam eklenen belge sayısı: {collection.count()}")

except Exception as e:
    print(f"\n❌ HATA: Veriler ChromaDB'ye eklenirken bir sorun oluştu: {e}")