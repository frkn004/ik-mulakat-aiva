# AIVA Mülakat Asistanı / AIVA Interview Assistant

## 🇹🇷 Türkçe

### Proje Hakkında
AIVA Mülakat Asistanı, yapay zeka destekli bir mülakat yönetim sistemidir. Sistem, adaylarla gerçek zamanlı sesli görüşme yapabilir, yanıtları değerlendirebilir ve detaylı raporlar oluşturabilir.

### Özellikler
- 🎙️ Gerçek zamanlı ses tanıma ve yanıt verme
- 🤖 GPT-4 destekli mülakat yönetimi
- 📊 Otomatik değerlendirme ve raporlama
- 🌐 Webhook entegrasyonu
- 📝 PDF rapor oluşturma
- 🔄 Çoklu dil desteği (Türkçe/İngilizce)

### Kullanım Senaryoları

#### 1. Manuel Mülakat Oluşturma
```bash
# Endpoint: POST /create_interview
{
    "candidate_name": "Aday Adı",
    "position": "Pozisyon",
    "requirements": ["Gereksinim 1", "Gereksinim 2"],
    "custom_questions": ["Soru 1", "Soru 2"]
}
```

#### 2. Webhook ile Mülakat Oluşturma
```bash
# Endpoint: POST /webhook/interview
{
    "adSoyad": "Aday Adı",
    "mail": "aday@email.com",
    "isIlaniPozisyonu": "Pozisyon",
    "isIlaniGereksinimleri": ["Gereksinim 1", "Gereksinim 2"],
    "mulakatSorulari": ["Soru 1", "Soru 2"]
}
```

### Mülakat Süreci
1. **Başlatma**
   - Manuel oluşturma veya webhook ile mülakat kodu oluşturulur
   - Sistem benzersiz bir mülakat kodu ve URL üretir

2. **Mülakat**
   - Aday, verilen URL üzerinden mülakata katılır
   - Sistem soruları sırayla sorar ve sesli yanıtlar alır
   - GPT-4 yanıtları analiz eder ve değerlendirir

3. **Raporlama**
   - Mülakat sonunda otomatik PDF raporu oluşturulur
   - Rapor, belirlenen klasöre kaydedilir
   - Webhook ile entegre sistemlere bildirim gönderilir

### Rapor İçeriği
- Aday bilgileri
- Pozisyon gereksinimleri
- Soru-cevap dökümü
- Teknik değerlendirme (100 üzerinden)
- İletişim becerileri değerlendirmesi
- Problem çözme yeteneği analizi
- Genel değerlendirme ve tavsiyeler

### Teknik Gereksinimler
- Python 3.8+
- Flask
- OpenAI API
- Google Cloud Speech-to-Text
- FFmpeg

### Kurulum
```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Çevresel değişkenleri ayarla
cp .env.example .env
# .env dosyasını düzenle

# Uygulamayı başlat
python app.py
```

## 🇬🇧 English

### About
AIVA Interview Assistant is an AI-powered interview management system. The system can conduct real-time voice interviews with candidates, evaluate responses, and generate detailed reports.

### Features
- 🎙️ Real-time speech recognition and response
- 🤖 GPT-4 powered interview management
- 📊 Automatic evaluation and reporting
- 🌐 Webhook integration
- 📝 PDF report generation
- 🔄 Multi-language support (Turkish/English)

### Usage Scenarios

#### 1. Manual Interview Creation
```bash
# Endpoint: POST /create_interview
{
    "candidate_name": "Candidate Name",
    "position": "Position",
    "requirements": ["Requirement 1", "Requirement 2"],
    "custom_questions": ["Question 1", "Question 2"]
}
```

#### 2. Interview Creation via Webhook
```bash
# Endpoint: POST /webhook/interview
{
    "adSoyad": "Candidate Name",
    "mail": "candidate@email.com",
    "isIlaniPozisyonu": "Position",
    "isIlaniGereksinimleri": ["Requirement 1", "Requirement 2"],
    "mulakatSorulari": ["Question 1", "Question 2"]
}
```

### Interview Process
1. **Initialization**
   - Interview code is generated manually or via webhook
   - System generates a unique interview code and URL

2. **Interview**
   - Candidate joins via provided URL
   - System asks questions sequentially and receives voice responses
   - GPT-4 analyzes and evaluates responses

3. **Reporting**
   - Automatic PDF report generation at the end
   - Report is saved to designated folder
   - Notification sent to integrated systems via webhook

### Report Content
- Candidate information
- Position requirements
- Q&A transcript
- Technical evaluation (out of 100)
- Communication skills assessment
- Problem-solving ability analysis
- General evaluation and recommendations

### Technical Requirements
- Python 3.8+
- Flask
- OpenAI API
- Google Cloud Speech-to-Text
- FFmpeg

### Installation
```bash
# Install required packages
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file

# Start application
python app.py
```

### Environment Variables
```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_credentials.json
WEBHOOK_URL=your_webhook_url
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
```

### Directory Structure
```
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── templates/         # HTML templates
├── reports/          # Generated PDF reports
├── interviews/       # Interview JSON files
└── .env             # Environment variables
```

### API Documentation

#### Create Interview
```http
POST /create_interview
Content-Type: application/json

{
    "candidate_name": "John Doe",
    "position": "Software Developer",
    "requirements": [
        "Bachelor's degree in Computer Science",
        "3+ years experience in Python"
    ],
    "custom_questions": [
        "Tell us about your projects",
        "What is your experience with APIs?"
    ]
}
```

#### Webhook Integration
```http
POST /webhook/interview
Content-Type: application/json

{
    "adSoyad": "John Doe",
    "mail": "john@example.com",
    "isIlaniPozisyonu": "Software Developer",
    "isIlaniGereksinimleri": [
        "Bachelor's degree in Computer Science",
        "3+ years experience in Python"
    ],
    "mulakatSorulari": [
        "Tell us about your projects",
        "What is your experience with APIs?"
    ]
}
```

### Error Handling
- Detailed error logging
- User-friendly error messages
- Automatic retry mechanisms
- Graceful fallbacks

### Security Features
- Secure file handling
- API key protection
- Rate limiting
- Input validation

### Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

<div align="center">
  <p>Developed with ❤️ by AIVA Tech</p>
  <p>
    <a href="https://www.aivatech.io">Website</a> |
    <a href="https://github.com/aivatech">GitHub</a> |
    <a href="https://www.linkedin.com/company/aivatech">LinkedIn</a>
  </p>
</div>
