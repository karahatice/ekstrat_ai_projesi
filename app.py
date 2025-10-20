import streamlit as st
import os
import chromadb
from chromadb.config import Settings
from transformers import pipeline

# --- YAPILANDIRMA VE BAŞLANGIÇ ---

# Sayfa ayarları
st.set_page_config(
    page_title="Ekstratların Hücre Canlılığını ve Hücreye Etkilerini Gösteren Asistan",
    layout="centered"
)

# *** BASİT STİL ENJEKSİYONU ***
st.markdown("""
<style>
/* 1. Sayfa Arkaplanını Hot Pink Yapar */
.stApp {
    background-color: #FF69B4; 
}
/* 2. Ana Başlık Rengini Lawn Green Yapar */
h1 {
    color: #7CFC00; 
}
/* 3. Ana İçerik Alanının Arkaplanını (Okunabilirlik İçin) Yarı Saydam Yapar */
.main > div {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
}
            
/* 4. Kural Tabanlı Öneri İÇERİĞİNİN Font Stilini Değiştirme (st.info/st.warning içindeki metin) */
/* Bu, st.info ve st.warning bileşenlerinin içindeki metni hedef alır */
[data-testid="stAlert"] * {
    font-family: 'Aptos', sans-serif !important;
    font-size: 18px !important; /* Font boyutu büyütüldü */
    color: #FFFFFF !important; /* Beyaz yapıldı */
}
            
</style>
""", unsafe_allow_html=True)

st.title("🌿 Ekstratların Hücre Canlılığını ve Etkilerini Gösteren Asistan")
st.markdown("### Bir soru veya ekstrat durumu giriniz. 👇🏻")

# --- PROJE HAKKINDA KUTUSU ---
with st.expander("Proje Hakkında (ℹ️)"):
    st.markdown("""
    Bu uygulama, doğal ekstratların farklı hücre hatları üzerindeki etkilerini analiz etmek amacıyla oluşturulmuştur. 
    Sistem, kullanıcı sorusuna en uygun bilimsel deney verilerini (ChromaDB) çeker ve bu bilgileri kullanarak 
    Google'ın FLAN-T5 modeli aracılığıyla akıcı bir yanıt üretir. Ayrıca basit sorular için kural tabanlı bir 
    karşılaştırma önerisi de sunar.
    """)

# --- CHROMA VERİTABANI BAĞLANTISI ---
from chromadb import PersistentClient
try:
    client = PersistentClient(path="./chroma_db") 
    collection = client.get_or_create_collection("extracts")
except Exception as e:
    st.error(f"ChromaDB'ye bağlanılamadı veya veritabanı klasöründe sorun var: {e}")
    collection = None

# --- MODEL YÜKLEME VE ÖN BELLEKLEME ---
@st.cache_resource
def load_model():
    # st.info("Dil Modeli yükleniyor (Bu işlem biraz zaman alabilir)...")
    try:
        model_pipe = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
        )
        # st.success("Dil Modeli başarıyla yüklendi!")
        return model_pipe
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}. Lütfen internet bağlantınızı kontrol edin.")
        return None

model = load_model()

# --- KURAL TABANLI MİNİ SİSTEM ---

def rule_based_suggestion(ekstrat, hücre_canlılığı):
    """
    Ekstrat ve hücre canlılığı anahtar kelimelerine göre sabit kuralları kontrol eder.
    """
    ekstrat = ekstrat.lower()
    hücre_canlılığı = hücre_canlılığı.lower()
    
    # Kural 1: Jojoba ve HaCaT
    if "jojoba" in ekstrat and "hacat" in hücre_canlılığı:
        return "Jojoba bitkisinden elde edilmiş ektratın cilt hücreleri üzerinde yatıştırıcı ve nemlendirici etkisi olduğu bilinmektedir. HaCaT -keratinosit- hücrelerinde; hücre canlılığını desteklediği, oksidatif stres nedeniyle oluşmuş hasara faydalı olabileceği ve düşük miktarda uygulandığında toksik etki etmediği düşünülmektedir. Ayrıca cilt bariyerini desteklediği ve inflamatuvar yanıtın hafifletilmesine yardımcı olduğu bilinmektedir. Jojoba ekstratının bu faydaları antioksidan bileşikleri ve içeriğinde bulunan yağ asitlerinden dolayı olduğu düşünülmektedir."
    
    # Kural 2: Gül ve HDF
    elif "gül" in ekstrat and "hdf" in hücre_canlılığı:
        return "Gül ekstratı, içeriğindeki fenolik bileşikler ve uçucu yağlar sayesinde cilt hücrelerinde yenileyici ve antioksidan etkiler gösterebilir. HDF -insan dermal fibroblast- hücrelerinde, ekstratın hücre canlılığını desteklediğini, kolajen sentezini teşvik ettiği ve serbest radikallere karşı koruyucu rol üstlendiği düşünülmektedir. Ayrıca ekstratın cilt elastikliğini artırıcı etkisiyle ve hafif antiinflamatuvar özellikleriyle doku onarım süreçlerinde hücrelere yardımcı olabilmektedir."
    
    # Kural 3: Çay Ağacı ve SZ95
    elif "çay ağacı" in ekstrat and "sz95" in hücre_canlılığı:
        return "Çay ağacı ekstratı, antimikrobiyal ve antiinflamatuvar özellikleriyle bilinir. SZ95 sebum hücrelerinde, çay ağacı ekstratının sebum üretimini dengeleyici ve inflamasyonu azaltıcı etkiler gösterdiği düşünülmektedir. Ayrıca oksidatif stres kaynaklı hücre hasarını hafifletmesi ve ciltteki mikrobiyal dengenin korunmasına katkı sağlayabilmektedir. Bu etkiler, ekstratın içerdiği terpenoid bileşikler (özellikle terpinen-4-ol) ile ilişkilendirilmektedir."
    
    # Kural 4: Jojoba ve SZ95
    elif "jojoba" in ekstrat and "sz95" in hücre_canlılığı:
        return "Jojoba ekstratı, ciltteki sebum dengesini düzenleyici özellikleriyle bilinir. SZ95 sebum hücrelerinde, jojoba ekstratının lipid sentezini dengelediği, aşırı sebum üretimini azalttığı ve oksidatif stres kaynaklı hücre hasarına karşı koruma sağlayabildiği düşünülmektedir. Ayrıca yatıştırıcı ve nemlendirici etkileriyle cilt bariyerine de faydaları olabilmektedir."
    
    # Kural 5: Gül ve HaCaT (Sizin sorduğunuz kural)
    elif "gül" in ekstrat and "hacat" in hücre_canlılığı:
        return "Gül ekstratı, doğal antioksidan ve fenolik bileşikler açısından zengindir. HaCaT -keratinosit- hücrelerinde, gül ekstratının hücre canlılığını artırabileceği, UV kaynaklı oksidatif strese karşı koruyucu etki gösterebileceği ve ciltteki inflamatuvar yanıtı azaltabileceği düşünülmektedir. Ayrıca cilt yenilenmesini destekleyen yumuşatıcı ve dengeleyici etkiler gösterebilir."
    
    # Kural 6: Çay Ağacı ve NHDF
    elif "çay ağacı" in ekstrat and ("nhdf" in hücre_canlılığı or "hdf" in hücre_canlılığı):
        return "Çay ağacı ekstratı, antimikrobiyal ve antiinflamatuvar özellikleriyle öne çıkar. NHDF -insan dermal fibroblast- hücrelerinde, çay ağacı ekstratının hücre canlılığını koruduğu, oksidatif stres kaynaklı hasarı hafiflettiği ve kolajen sentezini destekleyebildiği düşünülmektedir. Ayrıca yara iyileşmesi ve doku onarımına katkı sağlayan düzenleyici etkiler gösterebilir."   
    
    else:
        return None # Doğrudan eşleşme bulunamazsa None döndür


def extract_keywords_from_query(query):
    """
    Kullanıcı sorgusundan ekstrat ve hücre hattı anahtar kelimelerini ayıklar.
    """
    query = query.lower()
    ekstrat_keywords = ["jojoba", "gül", "çay ağacı"]
    hucre_keywords = ["hacat", "hdf", "sz95", "nhdf"]
    
    ekstrat_match = next((e for e in ekstrat_keywords if e in query), None)
    hucre_match = next((h for h in hucre_keywords if h in query), None)
    
    return ekstrat_match, hucre_match


# --- ANA UYGULAMA AKIŞI ---

query = st.text_input(
    "Örnek: 'Gül ekstratı, HaCaT hücre hattı canlılığını artırır mı?'", 
    key="query_input"
)

# *** Soruyu Çalıştır Butonu ***
if st.button("🔍 Soruyu Gönder"):
    
    # 1. Giriş Kontrolü
    if not query.strip(): 
        st.warning("Lütfen bir sorgu giriniz.")
        st.stop() 

    if model is None or collection is None:
        st.error("Uygulama, model veya veritabanı bağlantı hatası nedeniyle çalıştırılamıyor.")
        st.stop()
    
    try:
        with st.spinner("Veritabanından benzer veriler alınıyor ve model yanıtı oluşturuluyor..."):
            
            # 2. ChromaDB Sorgusu
            # (Eğer veritabanı boşsa docs ve metas boş gelecektir)
            results = collection.query(
                query_texts=[query], 
                n_results=6,
                include=['documents', 'metadatas'] 
            )
            
            docs = results["documents"][0] if results and results["documents"] else []
            metas = results["metadatas"][0] if results and results["metadatas"] else []
            
            context = "\n\n".join(docs)
            prompt = f"Soru: {query}\nBağlam (İlgili Bilgi Kaynağı): {context}\nYukarıdaki Bağlam'ı (Context) kullanarak Soruyu net ve akıcı Türkçe ile cevapla. Sadece Bağlam'da geçen bilgilere odaklan.\nCevap:"
            
            # 3. Modelden Yanıt Alma
            llm_result = model(prompt)
            output = llm_result[0]["generated_text"]
            
            # --- SONUÇLARI GÖSTERME ---
            
            # 4. Model Yanıtını Gösterme
            # st.success("🦾 Model Yanıtı:")
            # st.write(f"**{output}**") 
            #Web görünümünde Türkçesinde sıkıntı yaşadığım için ve gözüme estetik gelmediği için böyle yaptım.
            
            # 5. Kural Tabanlı Öneriyi Gösterme (DİREKT SORGUDAN PARS EDİLDİ)
            st.subheader("👩🏻‍🔬 Cellkrat'ın Önerisi:") #Asistana isim vermek istedim. "Cell+ekstrat" krat=Kratos'a da atıfta bulunmuş oldum.
            
            # Kullanıcı sorgusundan anahtar kelimeleri çıkar
            ekstrat_keyword, hucre_keyword = extract_keywords_from_query(query)
            
            if ekstrat_keyword and hucre_keyword:
                suggestion = rule_based_suggestion(ekstrat_keyword, hucre_keyword)
                
                if suggestion:
                    st.info(suggestion)
                else:
                    st.warning("Kural tabanlı sistemde bu kombinasyon için doğrudan bir eşleşme bulunamadı 😔.")
            else:
                st.warning("Kural tabanlı sistem için sorgudan yeterli ekstrat/hücre hattı anahtar kelimesi çıkarılamadı.")
                

    except Exception as e:
        st.error(f"Soru işlenirken beklenmedik bir hata oluştu: {e}")
        st.error("Lütfen terminal konsolunuzu kontrol edin ve veritabanı ile model bağlantılarını doğrulayın.")

# --- FOOTER (ALT BİLGİ) ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #333; font-size: 18px;'>Hazırlayan: Hatice Kara – 2025</p>", unsafe_allow_html=True)
# --- FOOTER SONU ---
   



    


