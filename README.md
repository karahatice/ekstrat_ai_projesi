# 🌿 Ekstratların Hücre Canlılığına Etkisini Gösteren Yapay Zeka Asistanı

Bu proje, çeşitli bitkisel ekstratların farklı hücre hatları üzerindeki **canlılık etkilerini** analiz etmek için geliştirilmiş bir yapay zeka tabanlı asistan sistemidir.

---

## Proje Amacı
Farklı ekstratların (örneğin **Gül**, **Jojoba**, **Çay ağacı**) insan hücre hatları (**HaCaT**, **HDF/NHDF**, **SZ95**) üzerindeki etkilerini içeren deneysel veriler üzerinden modelin anlamlı çıkarımlar yapabilmesini sağlamaktır.

---

##  Kullanılan Teknolojiler
-  **Python**
-  **Streamlit** – Arayüz
-  **ChromaDB** – Vektör veritabanı
-  **Sentence Transformers** – Anlamsal metin gömme (embedding)
-  **Hugging Face Transformers** – Dil modeli (T5 tabanlı)
-  **Pandas** – Veri işleme

---

## 🚀 Canlı Uygulamaya Ulaşın

Uygulamayı hemen denemek için aşağıdaki bağlantıyı kullanabilirsiniz:

[**Ekstrat AI Uygulamasını Hemen Dene**](https://ekstrataiprojesi-j2qgveeyidzmfqq9cxvhsr.streamlit.app/)

##  Kurulum ve Çalıştırma

###  Depoyu klonla
```bash
git clone https://github.com/<kullanici_adin>/<repo_adin>.git
cd <repo_adin>

## Sanal Ortam Oluşturma ve Etkinleştirme
python -m venv venv
venv\Scripts\activate      # Windows için
source venv/bin/activate   # Mac/Linux için

## Gerekli Kütüphaneleri Yükleme
pip install -r requirements.txt

## Embedding Oluşturma
python build_embeddings.py

## Uygulamayı Çalıştırma
streamlit run app.py
Tarayıcıda otomatik olarak açılmazsa şu adrese git:
👉 http://localhost:8501

🧪 Örnek Sorgular:
Jojoba HaCaT canlılığını artırıyor mu?
Jojoba ekstratının Keratinosit (HaCaT) hücre proliferasyonu üzerindeki etkisi nedir?
Jojoba SZ95 üzerinde ne gibi bir etki bırakıyor?
Sebosit (SZ95) hattında Jojoba ekstratının sebum dengeleme etkisine dair sonuçlar nelerdir?
Gül HDF'ye iyi geliyor mu?
Gül ekstratının dermal fibroblast (HDF) hücre hattında kollajen üretimine etkisi hakkında bilgi verir misin?
Gül ekstratı HaCaT canlılığını nasıl etkiler?
HaCaT hücrelerinde gül ekstratının antioksidan aktivitesine dair kural tabanlı öneri nedir?
Çay Ağacı SZ95 için toksik mi?
Çay Ağacı ekstratının Sebosit (SZ95) hücreleri üzerindeki sitotoksik eşiği nedir?
Çay Ağacı Fibroblastları (NHDF) nasıl etkiler?
Çay Ağacı'nın enflamasyonu azaltan etkisi NHDF hücreleri üzerinde nasıl gözlemlenmiştir?
!! Giriş projesi olduğu için bu sorular dışı sorulara yanıt vermemektedir. Lütfen bu soruları kullanınız.

👩🏻‍🔬 Hazırlayan: Hatice Kara

   


