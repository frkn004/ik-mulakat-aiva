import os
import json
import uuid
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_unique_code():
    """8 karakterlik benzersiz bir kod üretir."""
    return str(uuid.uuid4())[:8].upper()

def generate_interview_questions(position):
    """Belirtilen pozisyon için mülakat soruları üretir."""
    try:
        # OpenAI API'yi kullanarak sorular üret
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Sen bir insan kaynakları uzmanısın. Verilen pozisyon için uygun mülakat soruları hazırlayacaksın."
                },
                {
                    "role": "user",
                    "content": f"'{position}' pozisyonu için 5 adet mülakat sorusu hazırla. Her soru teknik bilgi ve deneyimi ölçmeye yönelik olmalı."
                }
            ]
        )

        # API yanıtından soruları ayıkla
        questions_text = response.choices[0].message.content
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        return questions[:5]  # En fazla 5 soru al
    except Exception as e:
        print(f"Soru üretme hatası: {str(e)}")
        return None

def create_interview(candidate_name, position):
    """Yeni bir mülakat oluşturur ve bilgilerini kaydeder."""
    try:
        # Benzersiz kod üret
        code = generate_unique_code()
        
        # Sorular üret
        questions = generate_interview_questions(position)
        if not questions:
            raise Exception("Mülakat soruları oluşturulamadı")

        # Mülakat verilerini hazırla
        interview_data = {
            'code': code,
            'candidate_name': candidate_name,
            'position': position,
            'questions': questions,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }

        # Verileri dosyaya kaydet
        interviews_dir = 'interviews'
        os.makedirs(interviews_dir, exist_ok=True)
        
        with open(os.path.join(interviews_dir, f'{code}.json'), 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)

        return code
    except Exception as e:
        print(f"Mülakat oluşturma hatası: {str(e)}")
        raise

def get_interview_by_code(code):
    """Mülakat koduna göre mülakat bilgilerini getirir."""
    try:
        file_path = os.path.join('interviews', f'{code}.json')
        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Mülakat bilgisi getirme hatası: {str(e)}")
        return None

def update_interview_status(code, status):
    """Mülakat durumunu günceller."""
    try:
        file_path = os.path.join('interviews', f'{code}.json')
        if not os.path.exists(file_path):
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)

        interview_data['status'] = status
        interview_data['updated_at'] = datetime.now().isoformat()

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Mülakat durumu güncelleme hatası: {str(e)}")
        return False 