import streamlit as st
import pickle
import re
from pythainlp.util import normalize

# โหลดโมเดล LogisticRegression
model_logis = pickle.load(open('model_logis-66130701712.pkl', 'rb'))

# โหลด TfidfVectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer-66130701712.pkl', 'rb'))

# ฟังก์ชันทำความสะอาดข้อความ
def TextClean(text):
  text = re.sub('<[^>]*>', '', text)
  emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
  text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
  text = normalize(text)
  return text

# ตั้งชื่อหัวข้อหน้าเว็บ
st.title("Text Sentiment Prediction Using Logistic Regression Model Created By Siwayu Seeyangnok ID: 66130701712")

def main():
  # สร้าง Sidebar สำหรับรับข้อความ
  text = st.text_input("ป้อนข้อความของคุณ")

  # ทำความสะอาดข้อความ
  text = TextClean(text)

  # แปลงข้อความให้เป็นเวกเตอร์ TF-IDF
  X_new_tfidf = tfidf_vectorizer.transform([text])

  # ทำนายผล
  prediction = model_logis.predict(X_new_tfidf)

  # แสดงผลลัพธ์
  if prediction == 'pos':
    st.success("ความคิดเห็นเชิงบวก")
  else:
    st.error("ความคิดเห็นเชิงลบ")

if __name__ == "__main__":
  main()
